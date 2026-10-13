"""Convert audited proposals into a separately declared, non-human experiment.

The official 144-crop dataset is read-only. This script does not create any
approved label, evaluation target, formal benchmark, or human-review claim.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
import zipfile

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if not (REPO/'learning').is_dir():
    REPO = Path.cwd()
if not (REPO/'learning').is_dir():
    raise RuntimeError('Run from the repository root or place this script under its scripts directory')
sys.path.insert(0, str(REPO))
from learning.annotations import annotation_revision, validate_annotation, rasterize_annotation
from learning.dataset import canonical_bytes, sha256_file, load_dataset_manifest, ContourDataset

RUN_ID = 'ai-weak-usgs-20261014-v1'


def rectangles(mask, block=4, conservative=True):
    """Native-pixel polygon cover; unknown cover always contains every pixel.

    Coarse blocks bound polygon allocation. Unknown uses any-of-block;
    positive non-contour regions use all-of-block. No uncertainty is cleared.
    """
    mask = np.asarray(mask, dtype=bool)
    values = mask.reshape(512//block, block, 512//block, block)
    grid = values.any(axis=(1, 3)) if conservative else values.all(axis=(1, 3))
    active, result = {}, []
    def finish(interval, start, stop):
        left, right = (int(v) for v in interval)
        x0, x1, y0, y1 = left*block-.5, right*block-.5, start*block-.5, stop*block-.5
        result.append([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    for y, row in enumerate(grid):
        switches = np.flatnonzero(np.r_[False, row, False][1:] != np.r_[False, row, False][:-1])
        intervals = list(zip(switches[::2], switches[1::2]))
        current = {interval: active.get(interval, y) for interval in intervals}
        for interval, start in active.items():
            if interval not in current:
                finish(interval, start, y)
        active = current
    for interval, start in active.items():
        finish(interval, start, len(grid))
    if len(result) > 4096:
        if block >= 16:
            raise RuntimeError('native uncertainty cover exceeds allocation limit')
        return rectangles(mask, block=block*2, conservative=conservative)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--drafts', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--archive', required=True, type=Path)
    args = parser.parse_args()
    original, drafts, root = args.dataset, args.drafts, args.output
    if (root/'manifest.json').exists() or args.archive.exists():
        raise RuntimeError('use a fresh experiment output/archive')
    root.mkdir(parents=True, exist_ok=True)
    for folder in ('images', 'sources', 'rights'):
        shutil.copytree(original/folder, root/folder, dirs_exist_ok=True)
    shutil.copyfile(original/'source-selection.json', root/'source-selection.json')
    (root/'annotations').mkdir(exist_ok=True)
    (root/'generators').mkdir(exist_ok=True)
    for filename in ('prepare_ai_drafts.py', 'build_weak_experiment.py'):
        source = Path(__file__).parent/filename
        if source.is_file():
            shutil.copyfile(source, root/'generators'/filename)
    data = json.loads((original/'manifest.json').read_text())
    source_manifest_sha = sha256_file(original/'manifest.json')
    draft_manifest = json.loads((drafts/'draft-manifest.json').read_text())
    if draft_manifest['source_dataset_manifest_sha256'] != source_manifest_sha:
        raise RuntimeError('drafts do not match the immutable source dataset')
    draft_map = {s['id']: s for s in draft_manifest['samples']}
    statistics = []
    for sample in data['samples']:
        proposal = draft_map[sample['id']]
        annotation_path = drafts/proposal['annotation']
        mask_path = drafts/proposal['masks']
        if sha256_file(annotation_path) != proposal['annotation_sha256'] or sha256_file(mask_path) != proposal['masks_sha256']:
            raise RuntimeError('AI proposal/mask bytes changed after final preparation')
        annotation = json.loads(annotation_path.read_text())
        masks = np.load(mask_path)
        candidate_valid = (masks['contour_ink'] | masks['noncontour_ink'] | masks['known_background']) & ~masks['ambiguous_ink']
        unknown = ~candidate_valid
        annotation.update(annotation_status='machine_labeled', review_level='automated_weak_labels',
                          human_review_required=True, reviewer_id=None,
                          all_visible_contours_confirmed=False, publication_ranking_eligible=False,
                          evaluation=None, case_difficulty='unassessed',
                          reviewed=[[[-.5, -.5], [511.5, -.5], [511.5, 511.5], [-.5, 511.5]]],
                          ignore=rectangles(unknown, conservative=True),
                          other_ink_regions=rectangles(masks['noncontour_ink'], conservative=False),
                          parent_revision=annotation['revision'],
                          weak_label_provenance=dict(method=draft_manifest['method'],
                              code_sha256=draft_manifest['code_sha256'], run_id=RUN_ID,
                              source_dataset_manifest_sha256=source_manifest_sha,
                              draft_manifest_sha256=sha256_file(drafts/'draft-manifest.json'),
                              converter_code_sha256=sha256_file(Path(__file__)),
                              candidate_masks_sha256=proposal['masks_sha256'],
                              uncertainty_cover='4px any-of-block rectangles; adaptive upward if allocation cap',
                              processed_extent_is_not_human_reviewed=True,
                              annotations_not_exhaustive=True,
                              contour_support_derived_from='source-anchored observed Ink + color OD; no gap bridges'))
        annotation['revision'] = annotation_revision(annotation)
        validate_annotation(annotation, sample=sample)
        targets = rasterize_annotation(annotation)
        if np.any(targets['valid_mask'] & unknown):
            raise RuntimeError(f"{sample['id']}: uncertainty entered supervised loss")
        if not targets['valid_mask'].any():
            raise RuntimeError(f"{sample['id']}: no conservative supervised pixels")
        path = root/'annotations'/f"{sample['id']}-machine-labeled.json"
        path.write_bytes(canonical_bytes(annotation))
        sample['annotation'] = dict(path=str(path.relative_to(root)), sha256=sha256_file(path),
                                    status='machine_labeled', revision=annotation['revision'])
        sample['difficulty'] = 'unassessed'
        for key in ('reference_file', 'reference_sha256', 'prompt'):
            sample.pop(key, None)
        row = dict(sample_id=sample['id'], split=sample['split'],
                   known_candidate_pixels=int(candidate_valid.sum()), supervised_pixels=int(targets['valid_mask'].sum()),
                   positive_contour_pixels=int((targets['centerline']>.5).sum()),
                   positive_risk_pixels=int((targets['other_ink']>.5).sum()),
                   ignored_pixels=int(targets['ignore_mask'].sum()),
                   ignored_polygons=len(annotation['ignore']),
                   monochrome_semantics_unresolved=annotation['proposal_provenance']['monochrome_semantics_unresolved'])
        statistics.append(row)
        print(sample['id'], row['supervised_pixels'], row['positive_contour_pixels'], flush=True)
    data.update(dataset_id='usgs-contour-ai-weak-16x144-v1',
                review_status='machine_labeled_requires_human_review',
                publication_ranking_eligible=False,
                parent_manifest_sha256=source_manifest_sha,
                weak_label_experiment=dict(enabled=True, human_review_required=True,
                    review_level='automated_weak_labels', run_id=RUN_ID,
                    formal_holdout_evaluation_allowed=False,
                    official_pilot_source_dataset_id='usgs-contour-pilot-16x144-v1'))
    manifest_path = root/'manifest.json'
    manifest_path.write_bytes(canonical_bytes(data))
    load_dataset_manifest(manifest_path)
    with (root/'weak-mask-statistics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(statistics[0]))
        writer.writeheader(); writer.writerows(statistics)
    note = ('AI WEAK LABEL EXPERIMENT. No annotation is human approved.\n'
            'Train/validation require explicit allow_weak_labels=True.\n'
            'Frozen calibration/holdout images are not used for training or checkpoint choice.\n'
            'No formal pilot performance claim or human evaluation target exists.\n'
            'Monochrome ink in sheets03/09/15 is wholly uncertain; only conservative paper background is supervised.\n'
            'reviewed records the machine-processed extent and does not mean human review.\n')
    (root/'HUMAN_REVIEW_REQUIRED.txt').write_text(note)
    with zipfile.ZipFile(args.archive, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob('*')):
            if path.is_file():
                archive.write(path, str(path.relative_to(root)))
    report = dict(run_id=RUN_ID, dataset_manifest_sha256=sha256_file(manifest_path),
                  archive=str(args.archive), archive_sha256=sha256_file(args.archive),
                  archive_bytes=args.archive.stat().st_size,
                  samples=144, human_approved_samples=0,
                  split_counts=data['split_policy']['sample_counts'],
                  original_dataset_untouched=sha256_file(original/'manifest.json')==source_manifest_sha,
                  uncertainty_pixels_in_loss=0,
                  sum_supervised_pixels=sum(r['supervised_pixels'] for r in statistics),
                  sum_positive_contour_pixels=sum(r['positive_contour_pixels'] for r in statistics))
    (drafts/'weak-experiment-report.json').write_bytes(canonical_bytes(report))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
