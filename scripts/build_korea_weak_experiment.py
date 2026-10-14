"""Add machine-labelled Korean monochrome sheets to the USGS weak experiment.

Two explicit stages keep source assignment frozen before any crop exists:

  select  parse a saved catalogue page, screen a seeded candidate pool and
          freeze 16 black-ink AMS L751 sheets with their splits
  build   download the frozen sheets, crop them, propose conservative
          monochrome contour/non-contour/unknown labels and merge them with
          an existing USGS weak-label dataset into a new archive

No label is human approved. Unresolved black ink, junctions and touching
strokes are excluded from the loss instead of being guessed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import time
from urllib.parse import quote
import zipfile

import numpy as np
import cv2
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from learning.annotations import annotation_revision, rasterize_annotation, validate_annotation
from learning.dataset import canonical_bytes, image_pixel_sha256, load_dataset_manifest, sha256_file
from scripts.build_weak_experiment import rectangles
from scripts.prepare_ai_drafts import paths_from_skeleton

Image.MAX_IMAGE_PIXELS = 400_000_000
CATALOG_URL = 'https://maps.lib.utexas.edu/maps/ams/korea_50k/'
RIGHTS_URL = 'https://maps.lib.utexas.edu/maps/faq.html'
RIGHTS_TEXT = (
    'University of Texas Libraries, Perry-Castaneda Library Map Collection FAQ, '
    '"Are the maps copyrighted?" (retrieved 2026-10-14):\n'
    '"Most of the maps scanned by the University of Texas Libraries and served from '
    'this web site are in the public domain. No permissions are needed to copy them. '
    'You may download them and use them as you wish. We appreciate credit to '
    '"University of Texas Libraries" as the source of the scanned images. A few maps '
    'are copyrighted, and are clearly marked as such."\n'
    'Series: Korea 1:50,000, Series L751, U.S. Army Map Service, 1945-. '
    'Works of the U.S. Government. Credit: University of Texas Libraries.\n')
SELECTION_SEED = 'archaeotrace-korea-ams-v1'
RUN_ID = 'ai-weak-usgs-korea-20261014-v1'
DATASET_ID = 'usgs-ams-korea-contour-ai-weak-v1'
POOL_SIZE = 48
SPLIT_PLAN = (('train', 8, 12), ('validation', 2, 12), ('calibration', 2, 6), ('holdout', 4, 6))
# Map body of an AMS L751 sheet; the legend, glossary and collar lie outside.
BODY = dict(x=(0.20, 0.88), y=(0.08, 0.74))
METHOD = 'monochrome-stroke-geometry-draft/1'
CFG = dict(paper_closing_px=25, ink_seed_od=0.40, ink_weak_od=0.20,
           blue_od=0.18, blue_seed_chroma=45.0, blue_weak_chroma=22.0,
           grid_kernel_px=81, straight_min_px=110, straight_gap_px=4,
           straight_cover=0.92, straight_halo_px=2,
           glyph_max_px=26, glyph_max_skeleton=70, glyph_compact=0.45, dot_px=5,
           spur_px=8, min_contour_px=30, max_mean_width=3.4, max_p90_width=4.6,
           thick_width=5.5, twin_max_px=3.6, twin_std_px=0.7, twin_frac=0.7,
           straight_ratio=0.98, straight_ratio_min_px=40,
           junction_halo_px=1, ignore_halo_px=1, background_max_od=0.10,
           context_halo_px=64, min_contour_skeleton_px=1500,
           fallback_contour_skeleton_px=600, last_contour_skeleton_px=250,
           max_blue_fraction=0.25)
N8 = np.ones((3, 3), bool)


def neighbours(skeleton):
    return ndi.convolve(skeleton.astype(np.uint8), N8.astype(np.uint8), mode='constant') - skeleton


def prune_spurs(skeleton, length, passes=2):
    """Remove short end branches so scan bumps do not split long strokes."""
    skeleton = skeleton.copy()
    for _ in range(passes):
        junction_zone = ndi.binary_dilation(skeleton & (neighbours(skeleton) >= 3), N8)
        pieces, count = ndi.label(skeleton & ~junction_zone, N8)
        if not count:
            break
        ends = skeleton & (neighbours(skeleton) == 1)
        sizes = np.bincount(pieces.ravel(), minlength=count + 1)
        has_end = np.bincount(pieces[ends], minlength=count + 1) > 0
        touches = np.bincount(pieces[ndi.binary_dilation(junction_zone, N8) & (pieces > 0)],
                              minlength=count + 1) > 0
        drop = has_end & touches & (sizes < length)
        drop[0] = False
        if not drop.any():
            break
        skeleton = skeletonize(skeleton & ~drop[pieces])
    return skeleton


def straight_lines(skeleton, ink):
    """Grid, rail and other long straight print; never treated as contour."""
    k = CFG['grid_kernel_px']
    found = ndi.binary_opening(ink, np.ones((1, k), bool)) | ndi.binary_opening(ink, np.ones((k, 1), bool))
    found = ndi.binary_dilation(found, iterations=1).astype(np.uint8)
    lines = cv2.HoughLinesP(skeleton.astype(np.uint8) * 255, 1, np.pi / 720, threshold=80,
                            minLineLength=CFG['straight_min_px'], maxLineGap=CFG['straight_gap_px'])
    near = ndi.binary_dilation(ink, iterations=1)
    for ax, ay, bx, by in ([] if lines is None else lines[:, 0]):
        count = int(np.hypot(bx - ax, by - ay)) + 1
        xs = np.linspace(ax, bx, count).round().astype(int)
        ys = np.linspace(ay, by, count).round().astype(int)
        if near[ys, xs].mean() >= CFG['straight_cover']:
            cv2.line(found, (int(ax), int(ay)), (int(bx), int(by)), 1,
                     thickness=2 * CFG['straight_halo_px'] + 1)
    return found.astype(bool) & near


def propose(context):
    """Conservative proposals for black-ink maps where colour cannot separate contours.

    Contours are long, thin, curved, unbranched strokes after spur pruning.
    Blue overprint, straight lines, compact glyphs and thick fills are
    non-contour ink. Everything else that is dark stays unknown.
    """
    rgb = np.asarray(context, dtype=np.float32)
    paper = np.stack([ndi.grey_closing(rgb[..., k], size=(CFG['paper_closing_px'],) * 2)
                      for k in range(3)], 2)
    od = np.maximum(0.0, np.log((paper + 5.0) / (rgb + 5.0)))
    mean_od = od.mean(2)
    chroma = rgb[..., 2] - 0.5 * (rgb[..., 0] + rgb[..., 1])
    blue_weak = (chroma > CFG['blue_weak_chroma']) & (od.max(2) > CFG['blue_od'])
    labels, count = ndi.label(blue_weak, N8)
    seeds = np.bincount(labels[chroma > CFG['blue_seed_chroma']], minlength=count + 1) > 0
    seeds[0] = False
    blue_ink = ndi.binary_dilation(seeds[labels], iterations=1)
    labels, count = ndi.label((mean_od > CFG['ink_weak_od']) & ~blue_ink, N8)
    seeds = np.bincount(labels[mean_od > CFG['ink_seed_od']], minlength=count + 1) > 0
    seeds[0] = False
    ink = seeds[labels]
    straight = straight_lines(skeletonize(ink), ink)
    other = blue_ink | straight
    width = 2.0 * ndi.distance_transform_edt(ink)
    rest = ink & ~straight
    labels, count = ndi.label(rest, N8)
    rest_skeleton = skeletonize(rest)
    rest_degree = neighbours(rest_skeleton)
    skeleton_length = np.bincount(labels[rest_skeleton], minlength=count + 1)
    glyph = np.zeros(count + 1, bool)
    fragment = np.zeros(count + 1, bool)
    for index, box in enumerate(ndi.find_objects(labels), start=1):
        if box is None:
            continue
        h, w = box[0].stop - box[0].start, box[1].stop - box[1].start
        if max(h, w) > CFG['glyph_max_px'] or skeleton_length[index] > CFG['glyph_max_skeleton']:
            continue
        mine = labels[box] == index
        branched = bool((rest_skeleton[box] & mine & (rest_degree[box] >= 3)).any())
        holes = bool((ndi.binary_fill_holes(mine) & ~mine).any())
        compact = min(h, w) >= CFG['glyph_compact'] * max(h, w)
        # Glyphs and symbols are branched, looped, compact or dot-sized.
        # Simple open strokes may be broken contours, so they stay unknown.
        if branched or holes or compact or max(h, w) <= CFG['dot_px']:
            glyph[index] = True
        else:
            fragment[index] = True
    glyph_mask = glyph[labels]
    other |= glyph_mask
    skeleton = prune_spurs(rest_skeleton & ~glyph_mask & ~fragment[labels], CFG['spur_px'])
    junction = skeleton & (neighbours(skeleton) >= 3)
    branches, count = ndi.label(
        skeleton & ~ndi.binary_dilation(junction, iterations=CFG['junction_halo_px']), N8)
    contour = np.zeros(skeleton.shape, bool)
    thick = np.zeros(skeleton.shape, bool)
    for index, box in enumerate(ndi.find_objects(branches), start=1):
        if box is None:
            continue
        y0, y1 = max(0, box[0].start - 8), min(skeleton.shape[0], box[0].stop + 8)
        x0, x1 = max(0, box[1].start - 8), min(skeleton.shape[1], box[1].stop + 8)
        mine = branches[y0:y1, x0:x1] == index
        n = int(mine.sum())
        if n < 8:
            continue
        stroke = width[y0:y1, x0:x1][mine]
        others = skeleton[y0:y1, x0:x1] & ~ndi.binary_dilation(mine, iterations=1)
        distance = ndi.distance_transform_edt(~others)[mine] if others.any() else np.full(n, 99.0)
        twin = distance <= CFG['twin_max_px']
        # Double-line roads and rivers keep a near-constant small separation.
        is_twin = bool(twin.any() and twin.mean() >= CFG['twin_frac']
                       and np.std(distance[twin]) <= CFG['twin_std_px'])
        points = np.stack(np.nonzero(mine)[::-1], 1).astype(float)
        points -= points.mean(0)
        projection = points @ np.linalg.svd(points, full_matrices=False)[2][0]
        is_straight = (n >= CFG['straight_ratio_min_px']
                       and (projection.max() - projection.min()) / n >= CFG['straight_ratio'])
        if (n >= CFG['min_contour_px'] and stroke.mean() <= CFG['max_mean_width']
                and np.percentile(stroke, 90) <= CFG['max_p90_width'] and not is_twin and not is_straight):
            contour[y0:y1, x0:x1] |= mine
        elif np.median(stroke) >= CFG['thick_width']:
            thick[y0:y1, x0:x1] |= mine
    # Filled symbols, buildings and heavy lettering.
    other |= ink & ndi.binary_dilation(thick, iterations=3) & (width >= CFG['thick_width'] - 1)
    contour_ink = ink & ndi.binary_dilation(contour, iterations=2) & ~other
    unresolved = ink & ~contour_ink & ~other
    ambiguous = (ndi.binary_dilation(unresolved, iterations=CFG['ignore_halo_px'])
                 | ndi.binary_dilation(junction, iterations=3))
    contour &= ~ambiguous
    background = (mean_od < CFG['background_max_od']) & ~ndi.binary_dilation(ink | blue_ink, iterations=2)
    return dict(contour_ink=contour_ink & ~ambiguous, contour_centerline=contour,
                noncontour_ink=other & ~ambiguous, ambiguous_ink=ambiguous,
                known_background=background & ~ambiguous, blue_ink=blue_ink)


def overlay(rgb, masks):
    result = np.asarray(rgb, dtype=np.uint8).copy()
    for name, color in (('ambiguous_ink', (255, 155, 0)), ('noncontour_ink', (225, 45, 100)),
                        ('contour_centerline', (0, 200, 55))):
        mask = ndi.binary_dilation(masks[name]) if name == 'contour_centerline' else masks[name]
        result[mask] = (0.3 * result[mask] + 0.7 * np.array(color)).astype(np.uint8)
    return Image.fromarray(result)


def seeded_order(items, label):
    return sorted(items, key=lambda item: hashlib.sha256(f'{SELECTION_SEED}|{label}|{item}'.encode()).hexdigest())


def sheet_name(filename):
    stem = re.sub(r'^txu-pclmaps-oclc-6612107-', '', Path(filename).stem)
    match = re.fullmatch(r"(.+)-(\d{4})-([0-9ivx]+)", stem)
    if not match:
        raise ValueError('unexpected AMS L751 file name: ' + filename)
    place, number, quadrant = match.groups()
    roman = {'1': 'I', '2': 'II', '3': 'III', '4': 'IV'}.get(quadrant, quadrant.upper())
    return place.replace("'", '').title(), number, roman


def download(url, path):
    if not path.exists():
        temporary = path.with_name(path.name + '.part')
        subprocess.run(['curl', '-sfL', '--retry', '3', url, '-o', str(temporary)], check=True)
        temporary.rename(path)
    return path


def screen(path):
    """Coarse edition screen: black-ink land sheets only, colour editions out."""
    with Image.open(path) as image:
        small = np.asarray(image.convert('RGB').reduce(4), dtype=np.float32)
    h, w = small.shape[:2]
    body = small[int(h * BODY['y'][0]):int(h * BODY['y'][1]), int(w * BODY['x'][0]):int(w * BODY['x'][1])]
    luminance = body.mean(2)
    paper = np.percentile(luminance, 90)
    dark = luminance < paper - 45
    spread = body.max(2) - body.min(2)
    blue = body[..., 2] - 0.5 * (body[..., 0] + body[..., 1])
    coloured = dark & (spread > 45) & (blue < 20)
    return dict(dark_fraction=round(float(dark.mean()), 4),
                colour_ink_share=round(float(coloured.sum() / max(1, dark.sum())), 4))


def select(args):
    out = args.selection
    if out.exists():
        raise RuntimeError('selection already frozen: ' + str(out))
    html = args.catalog_html.read_bytes()
    names = sorted(set(re.findall(r'korea_50k/(txu-pclmaps-oclc-6612107-[^"/]+\.jpg)',
                                  html.decode('utf-8', 'replace'))))
    pool = sorted(random.Random(SELECTION_SEED).sample(names, POOL_SIZE))
    args.cache.mkdir(parents=True, exist_ok=True)
    screened = []
    for name in pool:
        path = download(CATALOG_URL + quote(name), args.cache / name)
        screened.append(dict(file=name, **screen(path)))
        print(name, screened[-1], flush=True)
    eligible = [s['file'] for s in screened
                if s['colour_ink_share'] < args.max_colour_share and s['dark_fraction'] >= args.min_dark_fraction]
    ordered = seeded_order(eligible, 'sheet')
    needed = sum(count for _, count, _ in SPLIT_PLAN)
    if len(ordered) < needed:
        raise RuntimeError(f'only {len(ordered)} eligible sheets')
    sources, cursor = [], 0
    for split, count, crops in SPLIT_PLAN:
        for name in ordered[cursor:cursor + count]:
            place, number, quadrant = sheet_name(name)
            quadrangle = f'{place} {number} {quadrant}'
            sources.append(dict(id=f'kr-{len(sources) + 1:02d}', file=name, split=split, crop_count=crops,
                                quadrangle_name=quadrangle, state='kr-ams-l751',
                                family_id='kr-ams-l751-' + re.sub(r'[^a-z0-9]+', '-', quadrangle.lower()).strip('-'),
                                source_url=CATALOG_URL + quote(name)))
        cursor += count
    selection = dict(schema_version='archaeotrace-korea-source-selection/1', seed=SELECTION_SEED,
                     catalog_url=CATALOG_URL, catalog_sha256=hashlib.sha256(html).hexdigest(),
                     candidate_pool=dict(size=POOL_SIZE,
                                         method='random.Random(seed).sample(sorted catalogue jpg links)'),
                     screen=dict(rule='colour_ink_share < %.2f and dark_fraction >= %.3f on a 1/4 preview '
                                      'of the map body' % (args.max_colour_share, args.min_dark_fraction),
                                 body=BODY, results=screened),
                     assignment='SHA256(seed|sheet|file) order; splits filled train, validation, '
                                'calibration, holdout',
                     split_plan=[dict(split=s, sheets=n, crops_per_sheet=c) for s, n, c in SPLIT_PLAN],
                     sources=sources)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(canonical_bytes(selection))
    print(json.dumps([(s['id'], s['split'], s['quadrangle_name']) for s in sources], indent=1))


def crop_candidates(width, height, source_id):
    """Stratified jittered 6x6 grid inside the map body, in seeded order."""
    x0, x1 = (int(width * f) for f in BODY['x'])
    y0, y1 = (int(height * f) for f in BODY['y'])
    rng = random.Random(f'{SELECTION_SEED}|crops|{source_id}')
    cell_w, cell_h = (x1 - x0 - 512) / 6, (y1 - y0 - 512) / 6
    candidates = [(int(x0 + (column + rng.random()) * cell_w), int(y0 + (row + rng.random()) * cell_h))
                  for row in range(6) for column in range(6)]
    rng.shuffle(candidates)
    return candidates


def overlaps(box, chosen):
    # Crops of one sheet share a split, so partial overlap cannot leak data.
    return any(abs(box[0] - x) < 256 and abs(box[1] - y) < 256 for x, y in chosen)


def build(args):
    selection = json.loads(args.selection.read_text())
    usgs = load_dataset_manifest(args.usgs_weak / 'manifest.json')
    root = args.output
    if (root / 'manifest.json').exists() or args.archive.exists():
        raise RuntimeError('use a fresh output directory and archive')
    for folder in ('images', 'annotations', 'rights', 'generators', 'audit/overlays', 'audit/contact-sheets'):
        (root / folder).mkdir(parents=True, exist_ok=True)
    for folder in ('images', 'annotations', 'rights'):
        shutil.copytree(args.usgs_weak / folder, root / folder, dirs_exist_ok=True)
    for script in ('build_korea_weak_experiment.py', 'build_weak_experiment.py', 'prepare_ai_drafts.py'):
        shutil.copyfile(Path(__file__).parent / script, root / 'generators' / script)
    rights = root / 'rights' / 'pcl-ams-korea-rights.txt'
    rights.write_text(RIGHTS_TEXT)
    code_sha = sha256_file(Path(__file__))
    import scipy, skimage, PIL
    environment = dict(python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__,
                       skimage=skimage.__version__, pillow=PIL.__version__, opencv=cv2.__version__)
    created = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    data = json.loads(json.dumps(usgs.payload))
    for source in data['sources']:
        # Only the immutable URL/hash travels; the training archive stays small.
        source.pop('source_file', None)
    korean_sources, korean_samples, statistics = [], [], []
    args.cache.mkdir(parents=True, exist_ok=True)
    halo = CFG['context_halo_px']
    for entry in selection['sources']:
        path = download(entry['source_url'], args.cache / entry['file'])
        with Image.open(path) as opened:
            sheet = opened.convert('RGB')
        width, height = sheet.size
        chosen, montage_tiles = [], []
        for threshold in (CFG['min_contour_skeleton_px'], CFG['fallback_contour_skeleton_px'], CFG['last_contour_skeleton_px']):
            for x, y in crop_candidates(width, height, entry['id']):
                if len(chosen) == entry['crop_count']:
                    break
                if overlaps((x, y), [c[:2] for c in chosen]):
                    continue
                context = np.asarray(sheet.crop((x - halo, y - halo, x + 512 + halo, y + 512 + halo)))
                masks = {k: v[halo:halo + 512, halo:halo + 512] for k, v in propose(context).items()}
                if (masks['contour_centerline'].sum() < threshold
                        or masks['blue_ink'].mean() > CFG['max_blue_fraction']):
                    continue
                chosen.append((x, y, masks))
            if len(chosen) == entry['crop_count']:
                break
        # Sheets stay frozen; a coastal sheet may supply fewer crops, never
        # fewer than half, and the shortfall is recorded instead of hidden.
        if len(chosen) < (entry['crop_count'] + 1) // 2:
            raise RuntimeError(f"{entry['id']}: only {len(chosen)} usable crops")
        source = dict(id=entry['id'], family_id=entry['family_id'], split=entry['split'], aliases=[],
                      state=entry['state'], quadrangle_name=entry['quadrangle_name'],
                      title=f"AMS Korea 1:50,000 Series L751, {entry['quadrangle_name']}",
                      publisher='U.S. Army Map Service', series='Korea 1:50,000, Series L751, 1945-',
                      scan_credit='University of Texas Libraries',
                      license='Public domain (U.S. Government work)',
                      rights_status='public_domain', rights_statement_url=RIGHTS_URL,
                      rights_snapshot='rights/pcl-ams-korea-rights.txt',
                      rights_snapshot_sha256=sha256_file(rights),
                      source_url=entry['source_url'], source_sha256=sha256_file(path),
                      source_bytes=path.stat().st_size, source_size=[width, height],
                      raster_size=dict(width=width, height=height), georeferenced=False,
                      ink_semantics='monochrome black print with blue romanized overprint',
                      provenance_id=dict(authority='maps.lib.utexas.edu', namespace='pcl-ams-korea-50k',
                                         value=Path(entry['file']).stem),
                      crop_count=len(chosen), planned_crop_count=entry['crop_count'])
        korean_sources.append(source)
        for index, (x, y, masks) in enumerate(sorted(chosen, key=lambda c: (c[1], c[0])), start=1):
            sid = f"{entry['id']}-crop-{index:02d}"
            crop = sheet.crop((x, y, x + 512, y + 512))
            image_path = root / 'images' / f'{sid}.png'
            crop.save(image_path, format='PNG', optimize=True)
            sample = dict(id=sid, source_id=entry['id'], split=entry['split'], image=f'images/{sid}.png',
                          image_sha256=sha256_file(image_path), image_pixel_sha256=image_pixel_sha256(image_path),
                          source_crop_xywh=[x, y, 512, 512], source_tile_origin_xy=[x, y],
                          geotransform=[float(x), 1.0, 0.0, float(-y), 0.0, -1.0], difficulty='unassessed')
            # Unresolved strokes, their 1px halo and junctions are the only
            # unknowns; paper between classified strokes stays supervised.
            unknown = masks['ambiguous_ink']
            annotation = dict(
                schema_version='archaeotrace-learning-annotation/1', coordinate_space='pixel_xy',
                width=512, height=512, sample_id=sid, image_sha256=sample['image_sha256'],
                annotator_id='ai-monochrome-proposal-v1', reviewer_id=None,
                annotation_status='machine_labeled', review_level='automated_weak_labels',
                human_review_required=True, all_visible_contours_confirmed=False,
                publication_ranking_eligible=False, evaluation=None, case_difficulty='unassessed',
                contours=paths_from_skeleton(masks['contour_centerline'], 'contour'), other_ink=[],
                reviewed=[[[-.5, -.5], [511.5, -.5], [511.5, 511.5], [-.5, 511.5]]],
                ignore=rectangles(unknown, block=2, conservative=True),
                other_ink_regions=rectangles(masks['noncontour_ink'], conservative=False),
                proposal_provenance=dict(
                    method=METHOD, config=CFG, created_at_utc=created, source_id=entry['id'],
                    source_sha256=source['source_sha256'], source_crop_xywh=[x, y, 512, 512],
                    source_context_halo_px=halo, code_sha256=code_sha, generator_environment=environment,
                    no_missing_ink_bridges=True, human_approved=False),
                weak_label_provenance=dict(
                    method=METHOD, code_sha256=code_sha, run_id=RUN_ID,
                    uncertainty_cover='2px any-of-block rectangles; adaptive upward if allocation cap',
                    processed_extent_is_not_human_reviewed=True, annotations_not_exhaustive=True,
                    contour_support_derived_from='thin unbranched curved black strokes; no gap bridges'))
            annotation['revision'] = annotation_revision(annotation)
            validate_annotation(annotation, sample=sample)
            targets = rasterize_annotation(annotation)
            if np.any(targets['valid_mask'] & unknown):
                raise RuntimeError(f'{sid}: uncertainty entered supervised loss')
            annotation_path = root / 'annotations' / f'{sid}-machine-labeled.json'
            annotation_path.write_bytes(canonical_bytes(annotation))
            sample['annotation'] = dict(path=f'annotations/{sid}-machine-labeled.json',
                                        sha256=sha256_file(annotation_path), status='machine_labeled',
                                        revision=annotation['revision'])
            korean_samples.append(sample)
            rendered = overlay(crop, masks)
            rendered.save(root / 'audit' / 'overlays' / f'{sid}.png')
            montage_tiles.append((sid, crop, rendered))
            row = dict(sample_id=sid, source_id=entry['id'], split=entry['split'],
                       contour_paths=len(annotation['contours']),
                       supervised_pixels=int(targets['valid_mask'].sum()),
                       positive_contour_pixels=int((targets['centerline'] > .5).sum()),
                       positive_risk_pixels=int((targets['other_ink'] > .5).sum()),
                       ignored_pixels=int(targets['ignore_mask'].sum()))
            statistics.append(row)
            print(sid, row['positive_contour_pixels'], row['ignored_pixels'], flush=True)
        montage = Image.new('RGB', (1024, 544 * len(montage_tiles)), 'white')
        draw = ImageDraw.Draw(montage)
        for index, (sid, crop, rendered) in enumerate(montage_tiles):
            montage.paste(crop, (0, index * 544 + 24))
            montage.paste(rendered, (512, index * 544 + 24))
            draw.text((4, index * 544 + 4),
                      sid + '  original / AI DRAFT (green contour, magenta other, orange unknown)', fill='black')
        montage.save(root / 'audit' / 'contact-sheets' / f"{entry['id']}.png")
    combined = dict(schema_version='archaeotrace-learning-source-selection/1', seed=RUN_ID,
                    parents=dict(usgs_source_selection_sha256=usgs.payload['source_selection']['sha256'],
                                 korea_source_selection_sha256=sha256_file(args.selection)),
                    sources=[dict(id=s['id'], family_id=s['family_id'], split=s['split'],
                                  source_url=s['source_url']) for s in data['sources'] + korean_sources])
    (root / 'source-selection.json').write_bytes(canonical_bytes(combined))
    shutil.copyfile(args.selection, root / 'korea-source-selection.json')
    data['sources'] += korean_sources
    data['samples'] += korean_samples

    def counts(items):
        return {split: sum(1 for item in items if item['split'] == split)
                for split in ('train', 'validation', 'calibration', 'holdout')}

    data.update(
        dataset_id=DATASET_ID, seed=RUN_ID, parent_manifest_sha256=usgs.sha256,
        source_selection=dict(path='source-selection.json', sha256=sha256_file(root / 'source-selection.json')),
        split_policy=dict(assignment_unit='sheet_family', frozen=True,
                          source_counts=counts(data['sources']), sample_counts=counts(data['samples'])),
        review_status='machine_labeled_requires_human_review', publication_ranking_eligible=False,
        weak_label_experiment=dict(
            enabled=True, human_review_required=True, review_level='automated_weak_labels', run_id=RUN_ID,
            formal_holdout_evaluation_allowed=False,
            official_pilot_source_dataset_id='usgs-contour-pilot-16x144-v1',
            korea_monochrome_method=METHOD,
            korea_crop_selection='stratified-grid-jitter in map body; crops need >= %d contour-proposal '
                                 'skeleton px (fallbacks %d, %d)' % (CFG['min_contour_skeleton_px'],
                                                                     CFG['fallback_contour_skeleton_px'],
                                                                     CFG['last_contour_skeleton_px'])))
    manifest_path = root / 'manifest.json'
    manifest_path.write_bytes(canonical_bytes(data))
    load_dataset_manifest(manifest_path)
    with (root / 'audit' / 'korea-weak-statistics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(statistics[0]))
        writer.writeheader()
        writer.writerows(statistics)
    (root / 'HUMAN_REVIEW_REQUIRED.txt').write_text(
        'AI WEAK LABEL EXPERIMENT. No annotation is human approved.\n'
        'Korean AMS L751 labels come from stroke geometry on black print; unresolved ink is ignored.\n'
        'Train/validation require explicit allow_weak_labels=True.\n'
        'Calibration/holdout images are never used for training or checkpoint choice.\n')
    with zipfile.ZipFile(args.archive, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob('*')):
            if path.is_file() and 'audit' not in path.relative_to(root).parts:
                archive.write(path, str(path.relative_to(root)))
    report = dict(run_id=RUN_ID, dataset_id=DATASET_ID, dataset_manifest_sha256=sha256_file(manifest_path),
                  archive=str(args.archive), archive_sha256=sha256_file(args.archive),
                  archive_bytes=args.archive.stat().st_size, human_approved_samples=0,
                  split_counts=data['split_policy']['sample_counts'],
                  source_counts=data['split_policy']['source_counts'],
                  korea_samples=len(korean_samples),
                  korea_positive_contour_pixels=sum(r['positive_contour_pixels'] for r in statistics))
    (root / 'audit' / 'build-report.json').write_bytes(canonical_bytes(report))
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    first = commands.add_parser('select')
    first.add_argument('--catalog-html', required=True, type=Path)
    first.add_argument('--cache', required=True, type=Path)
    first.add_argument('--selection', required=True, type=Path)
    first.add_argument('--max-colour-share', type=float, default=0.10)
    first.add_argument('--min-dark-fraction', type=float, default=0.10)
    second = commands.add_parser('build')
    second.add_argument('--selection', required=True, type=Path)
    second.add_argument('--usgs-weak', required=True, type=Path)
    second.add_argument('--cache', required=True, type=Path)
    second.add_argument('--output', required=True, type=Path)
    second.add_argument('--archive', required=True, type=Path)
    args = parser.parse_args()
    select(args) if args.command == 'select' else build(args)


if __name__ == '__main__':
    main()
