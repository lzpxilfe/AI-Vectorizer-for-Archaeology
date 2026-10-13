"""Source-anchored, conservative color proposals; never human-approved labels.

Usage: test-env/bin/python prepare_ai_drafts.py --dataset PATH --output PATH
No manuscript validation, approved labels, or evaluation targets are created.
"""
from __future__ import annotations

import argparse
import csv
import io
import hashlib
import json
import shutil
from pathlib import Path
import sys
import time

import numpy as np
import cv2
from PIL import Image, ImageDraw
from PIL import ImageOps
import subprocess
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

REPO = Path(__file__).resolve().parents[1]
if not (REPO/'learning').is_dir():
    REPO = Path.cwd()
if not (REPO/'learning').is_dir():
    raise RuntimeError('Run from the repository root or place this script under its scripts directory')
sys.path.insert(0, str(REPO))
from learning.annotations import annotation_revision, validate_annotation
from learning.dataset import canonical_bytes, sha256_file
from ai_vectorizer.core.edge_detector import EdgeDetector

METHOD = 'native-paper-OD-color-draft/1'
MONOCHROME_SOURCES = {'sheet-03', 'sheet-09', 'sheet-15'}
CONFIG = dict(context_halo_px=64, paper_closing_px=31,
              brown_seed_blue_od=0.11, brown_seed_blue_minus_red=0.075,
              brown_seed_green_minus_red=0.035,
              brown_weak_blue_od=0.065, brown_weak_blue_minus_red=0.04,
              brown_weak_green_minus_red=0.018,
              grey_seed_od=0.18, grey_chroma_bound=0.13,
              min_seed_pixels=3, min_contour_component_skeleton_px=12,
              min_other_component_skeleton_px=4,
              background_max_od=0.055, ambiguity_halo_px=2,
              brown_relative_blue_chroma=.35, brown_relative_green_chroma=.20,
              straight_unknown_length_px=180, straight_unknown_radius_px=5,
              straight_unknown_max_gap_px=24,
              ocr_unknown_confidence=50, ocr_unknown_padding_px=3,
              ocr_max_area_fraction=.08, ocr_max_height_px=96, ocr_max_width_px=380,
              ocr_input_scale=2, ocr_engine='tesseract --psm 11')


def ocr_uncertainty(crop):
    """OCR creates ignore-only boxes; text and elevations are never targets."""
    image = ImageOps.autocontrast(crop.convert('L')).resize((1024, 1024))
    buffer = io.BytesIO(); image.save(buffer, format='PNG')
    result = subprocess.run(['tesseract', 'stdin', 'stdout',
                             '--psm', '11', 'tsv'], input=buffer.getvalue(),
                            capture_output=True, check=True, timeout=30)
    boxes, mask = [], np.zeros((512, 512), dtype=bool)
    for row in csv.DictReader(io.StringIO(result.stdout.decode()), delimiter='\t'):
        text = row.get('text', '').strip()
        if len(text) < 2 or float(row['conf']) < 50:
            continue
        x0 = max(0, int(row['left'])//2-3)
        y0 = max(0, int(row['top'])//2-3)
        x1 = min(512, (int(row['left'])+int(row['width'])+1)//2+3)
        y1 = min(512, (int(row['top'])+int(row['height'])+1)//2+3)
        if (x1-x0)*(y1-y0) > .08*512*512 or y1-y0 > 96 or x1-x0 > 380:
            continue
        mask[y0:y1, x0:x1] = True
        boxes.append(dict(text=text, confidence=float(row['conf']),
                          native_xyxy=[x0, y0, x1, y1], use='ignore_only'))
    return mask, boxes


def paths_from_skeleton(mask, prefix):
    points = set((int(x), int(y)) for y, x in zip(*np.nonzero(mask)))
    neighbors = {}
    for x, y in points:
        neighbors[(x, y)] = sorted((x+dx, y+dy)
            for dx in (-1, 0, 1) for dy in (-1, 0, 1)
            if (dx or dy) and (x+dx, y+dy) in points
            # Avoid artificial triangle junctions when two diagonal pixels
            # already share an orthogonal foreground neighbor.
            and not (dx and dy and ((x+dx, y) in points or (x, y+dy) in points)))
    visited, records = set(), []
    for start in sorted(points, key=lambda p: (len(neighbors[p]) == 2, p)):
        for other in neighbors[start]:
            edge = tuple(sorted((start, other)))
            if edge in visited:
                continue
            visited.add(edge)
            path = [start, other]
            previous, current = start, other
            while len(neighbors[current]) == 2 and len(path) < 100000:
                following = next(p for p in neighbors[current] if p != previous)
                edge = tuple(sorted((current, following)))
                if edge in visited:
                    break
                visited.add(edge)
                path.append(following)
                previous, current = current, following
            if len(path) >= 4:
                records.append({'id': f'{prefix}-{len(records):04d}',
                                'points': [[float(x), float(y)] for x, y in path]})
    if len(records) > 4096:
        raise RuntimeError('path proposal count exceeds annotation schema')
    return records


def seeded_components(weak, seeds):
    labels, count = ndi.label(weak, structure=np.ones((3, 3), dtype=bool))
    seed_counts = np.bincount(labels[seeds], minlength=count+1)
    accepted = seed_counts >= CONFIG['min_seed_pixels']
    accepted[0] = False
    return accepted[labels]


def filter_components(mask, minimum, suspect_glyph=False):
    labels, count = ndi.label(mask, structure=np.ones((3, 3), dtype=bool))
    skeleton = skeletonize(mask)
    lengths = np.bincount(labels[skeleton], minlength=count+1)
    keep = lengths >= minimum
    keep[0] = False
    if suspect_glyph:
        # Brown altitude digits are not separable from contours by color.
        # Compact components are exposed as ambiguous instead of labelled.
        for index, sl in enumerate(ndi.find_objects(labels), start=1):
            if sl is None or not keep[index]:
                continue
            h, w = sl[0].stop-sl[0].start, sl[1].stop-sl[1].start
            if 4 <= w <= 24 and 4 <= h <= 28 and lengths[index] < 90:
                keep[index] = False
    return keep[labels]


def propose(context, selection, monochrome, native_origin):
    rgb = np.asarray(context, dtype=np.float32)
    paper = np.stack([ndi.grey_closing(rgb[:, :, k], size=(31, 31))
                      for k in range(3)], axis=2)
    od = np.maximum(0., np.log((paper+5.)/(rgb+5.)))
    red, green, blue = np.moveaxis(od, -1, 0)
    max_od, min_od = od.max(axis=2), od.min(axis=2)
    if monochrome:
        contour = np.zeros(max_od.shape, dtype=bool)
        other = np.zeros(max_od.shape, dtype=bool)
    else:
        relative_chroma = (blue-red > .35*blue) & (green-red > .20*green)
        brown_seed = (blue > .11) & (blue-red > .075) & (green-red > .035) & (blue > .75*green) & relative_chroma
        brown_weak = (blue > .065) & (blue-red > .04) & (green-red > .018) & (blue > .65*green) & relative_chroma
        contour = seeded_components(brown_weak, brown_seed)
        contour = filter_components(contour, 12, suspect_glyph=True)
        # Black/blue print is non-contour only in color-separated sources.
        black = (min_od > .18) & (max_od-min_od < .13)
        cyan = (red > .13) & (red-blue > .07) & (green-blue > .015)
        non_brown = (blue-red < .03) | (green-red < .005)
        other_seed = (black | cyan) & non_brown & ~contour
        other_weak = (max_od > .075) & non_brown & ~contour
        other = seeded_components(other_weak, other_seed)
        other = filter_components(other, 4)
    ink = max_od > .07
    ambiguous = ink & ~contour & ~other
    ambiguous = ndi.binary_dilation(ambiguous, iterations=2)
    # Candidate contours adjacent to unresolved print are withheld, not bridged.
    # Keep the repository's directly observed, source-anchored Ink centerline.
    # Skeletonizing a color mask with scan noise creates many false branches.
    evidence = EdgeDetector.detect_ink_evidence(np.asarray(context), tile_origin=native_origin)
    unmatched_observed_ink = evidence.centerline & ~ndi.binary_dilation(contour | other)
    ambiguous |= ndi.binary_dilation(unmatched_observed_ink, iterations=2)
    # Grid, survey and road lines may share brown scan residuals. Long nearly
    # straight structures are withheld, even when a true contour also happens
    # to be straight. Filling a dashed corridor affects only ambiguity, never
    # creates positive ink or an artificial contour bridge.
    corridor = np.zeros(ink.shape, dtype=np.uint8)
    lines = cv2.HoughLinesP(evidence.centerline.astype(np.uint8)*255, 1,
                           np.pi/720, threshold=100, minLineLength=180,
                           maxLineGap=24)
    if lines is not None:
        for line in lines[:, 0]:
            ax, ay, bx, by = (int(v) for v in line)
            if np.hypot(bx-ax, by-ay) >= 180:
                cv2.line(corridor, (ax, ay), (bx, by), 1, thickness=11)
    ambiguous |= corridor.astype(bool)
    contour_center = evidence.centerline & ndi.binary_dilation(contour) & ~ambiguous
    other_center = evidence.centerline & ndi.binary_dilation(other) & ~ambiguous
    contour &= ~ambiguous
    other &= ~ambiguous
    contour, other, ambiguous = (a[selection].copy() for a in (contour, other, ambiguous))
    background = ((max_od < .055) & ~ndi.binary_dilation(ink, iterations=2))[selection].copy() & ~ambiguous
    contour_center = contour_center[selection].copy()
    other_center = other_center[selection].copy()
    return dict(contour_ink=contour, contour_centerline=contour_center,
                noncontour_ink=other, noncontour_centerline=other_center,
                ambiguous_ink=ambiguous, known_background=background,
                straight_unknown=corridor[selection].astype(bool))


def overlay(rgb, masks):
    result = np.asarray(rgb, dtype=np.uint8).copy()
    for name, color in [('ambiguous_ink', (255, 155, 0)),
                        ('noncontour_ink', (225, 45, 100)),
                        ('contour_centerline', (0, 200, 55))]:
        mask = masks[name]
        if name == 'contour_centerline':
            mask = ndi.binary_dilation(mask)
        result[mask] = (.35*result[mask]+.65*np.array(color)).astype(np.uint8)
    return Image.fromarray(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if shutil.which('tesseract') is None:
        raise RuntimeError('Tesseract is required for text uncertainty boxes; it cannot be silently skipped')
    import scipy, skimage, PIL
    environment = dict(python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__,
                       skimage=skimage.__version__, pillow=PIL.__version__, opencv=cv2.__version__,
                       tesseract=subprocess.run(['tesseract', '--version'], capture_output=True,
                                                check=True, text=True).stdout.splitlines()[0])
    root, out = args.dataset, args.output
    for folder in ('annotations', 'masks', 'overlays', 'contact-sheets'):
        (out/folder).mkdir(parents=True, exist_ok=True)
    dataset = json.loads((root/'manifest.json').read_text())
    statistics, samples = [], []
    sources = {s['id']: s for s in dataset['sources']}
    created = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    for source_id, source in sources.items():
        source_rgb = Image.open(root/source['source_file']).convert('RGB')
        cases = [s for s in dataset['samples'] if s['source_id'] == source_id]
        montage = Image.new('RGB', (4*512, ((len(cases)+1)//2)*544), 'white')
        draw = ImageDraw.Draw(montage)
        for index, sample in enumerate(cases):
            sid = sample['id']
            x, y, w, h = sample['source_crop_xywh']
            x0, y0 = max(0, x-64), max(0, y-64)
            x1, y1 = min(source_rgb.width, x+w+64), min(source_rgb.height, y+h+64)
            context = source_rgb.crop((x0, y0, x1, y1))
            selection = np.s_[y-y0:y-y0+h, x-x0:x-x0+w]
            crop = Image.open(root/sample['image']).convert('RGB')
            if not np.array_equal(np.asarray(context)[selection], np.asarray(crop)):
                raise RuntimeError(f'{sid}: source/native crop pixels mismatch')
            monochrome = source_id in MONOCHROME_SOURCES
            masks = propose(context, selection, monochrome, (x0, y0))
            text_mask, text_boxes = ocr_uncertainty(crop)
            masks['ocr_unknown'] = text_mask
            masks['ambiguous_ink'] |= text_mask
            for key in ('contour_ink', 'contour_centerline', 'noncontour_ink',
                        'noncontour_centerline', 'known_background'):
                masks[key] &= ~masks['ambiguous_ink']
            annotation = json.loads((root/sample['annotation']['path']).read_text())
            annotation.update(annotator_id='ai-color-proposal-v1', reviewer_id=None,
                annotation_status='draft', review_level='unreviewed_ai_draft',
                all_visible_contours_confirmed=False, publication_ranking_eligible=False,
                reviewed=[], ignore=[], evaluation=None, case_difficulty='unassessed',
                contours=paths_from_skeleton(masks['contour_centerline'], 'contour'),
                other_ink=paths_from_skeleton(masks['noncontour_centerline'], 'other'),
                other_ink_regions=[], parent_revision=sample['annotation']['revision'],
                proposal_provenance=dict(method=METHOD, config=CONFIG, created_at_utc=created,
                    source_id=source_id, source_sha256=source['source_sha256'],
                    source_crop_xywh=sample['source_crop_xywh'],
                    source_context_xywh=[x0, y0, x1-x0, y1-y0],
                    source_manifest_sha256=sha256_file(root/'manifest.json'),
                    code_sha256=sha256_file(Path(__file__)),
                    monochrome_semantics_unresolved=monochrome,
                    ocr_ignore_boxes=text_boxes,
                    generator_environment=environment,
                    no_missing_ink_bridges=True, human_approved=False))
            annotation['revision'] = annotation_revision(annotation)
            validate_annotation(annotation, sample=sample)
            ann_path = out/'annotations'/f'{sid}-ai-draft.json'
            ann_path.write_bytes(canonical_bytes(annotation))
            mask_path = out/'masks'/f'{sid}.npz'
            np.savez_compressed(mask_path, **masks)
            rendered = overlay(crop, masks)
            rendered.save(out/'overlays'/f'{sid}.png')
            cx, cy = (index%2)*1024, (index//2)*544
            montage.paste(crop, (cx, cy+24))
            montage.paste(rendered, (cx+512, cy+24))
            draw.text((cx+3, cy+4), sid+'  original / AI DRAFT overlay', fill='black')
            row = dict(sample_id=sid, source_id=source_id, split=sample['split'],
                       monochrome_semantics_unresolved=monochrome,
                       contour_paths=len(annotation['contours']), other_paths=len(annotation['other_ink']),
                       **{key+'_pixels': int(value.sum()) for key, value in masks.items()})
            statistics.append(row)
            samples.append(dict(id=sid, source_id=source_id, split=sample['split'],
                image=str((root/sample['image']).resolve()), image_sha256=sample['image_sha256'],
                source_crop_xywh=sample['source_crop_xywh'],
                annotation='annotations/'+ann_path.name, annotation_sha256=sha256_file(ann_path),
                masks='masks/'+mask_path.name, masks_sha256=sha256_file(mask_path),
                overlay='overlays/'+sid+'.png', statistics=row))
            print(sid, 'monochrome' if monochrome else 'color', row['contour_paths'], flush=True)
        montage.save(out/'contact-sheets'/f'{source_id}.png')
        source_rgb.close()
    with (out/'candidate-statistics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(statistics[0]))
        writer.writeheader(); writer.writerows(statistics)
    manifest = dict(schema_version='archaeotrace-ai-draft-preparation/1',
                    source_dataset_manifest=str((root/'manifest.json').resolve()),
                    source_dataset_manifest_sha256=sha256_file(root/'manifest.json'),
                    method=METHOD, config=CONFIG, environment=environment, code_sha256=sha256_file(Path(__file__)),
                    created_at_utc=created, human_approved_samples=0,
                    official_training_eligible=False, publication_ranking_eligible=False,
                    monochrome_sources=sorted(MONOCHROME_SOURCES),
                    legend=dict(green='contour center candidate', magenta='non-contour ink candidate',
                                orange='unresolved ink, withhold from any pseudo-label loss'),
                    samples=samples)
    (out/'draft-manifest.json').write_bytes(canonical_bytes(manifest))
    print('Complete:', len(samples), 'DRAFTS; human-approved:', 0, flush=True)


if __name__ == '__main__':
    main()
