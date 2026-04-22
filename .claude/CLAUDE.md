# CLAUDE.md

This project uses the [Propel](https://github.com/KevinBian107/propel) research workflow.
Skills, agents, and commands are in `.claude/` (installed via `propel init`).

## Project Overview

A PyQt5 desktop GUI for frame-accurate annotation of animal behavior in videos, inspired by ChronoViz. This repo (`TopoMimic-annotation`) is the standalone home of the fork; the upstream is `ericleonardis/annotation-gui`. Used as tooling for producing labeled rodent-behavior datasets in the broader topo-vnl / TopoMimic research pipeline.

Architecture (single main module + small supporting modules):
- `annotation_GUI.py` — entire app; three classes:
  - `BehaviorSegment` — time-range annotation with class-level color map. Segments are used for BOTH the behavior track and the direction track. `name` carries either a macro-behavior (Immobile / Rear / Turn / Walk / Groom) or a direction label (Left / Right / Straight).
  - `TimelineWidget` — custom `QWidget` for visualizing and editing segments. Renders a direction strip at the top + behavior rows below. Signals: `clicked_pos`, `dragging`, `segment_selected`, `segment_modified`, `segment_drag_started`. Also renders an auto-label overlay on behavior rows (unchanged from pre-refactor).
  - `AnnotatorGUI` — main `QMainWindow` that owns videos (each with `segments`, `direction_segments`, `auto_segments`), behaviors, hotkey state, undo/redo stacks, and both auto pipelines.
- `convert_labels.py` — transforms legacy 6-column CSV to the two-track 7-column schema `Video,Track,Label,Start_Time,End_Time,Start_Frame,End_Frame`
- `auto_labeler.py` — heuristic macro-behavior labeler (pure Python, no numpy)
- `auto_direction.py` — separate Left/Right/Straight pipeline on the same qpos
- `theme.py` — light theme (Fusion style + QPalette + QSS + `THEME` dict consumed by `TimelineWidget.paintEvent`)
- `scripts/calibrate_thresholds.py` — one-shot calibration from the converted two-track CSV → `auto_label_config.json`
- `docs/auto_labeling.md` — per-behavior rubric: what each rule looks for, features, conditions, failure modes
- `tests/` — pytest suite (`tests/conftest.py` handles sys.path + `ANNOTATION_GUI_NO_AUTOLOAD=1`)
- `environment.yaml` — conda env named `behavior_gui` (Python 3.10)
- `figures/` — `logo.png` and `demo.png` referenced from the top-level README
- **No `sample/` ships with this repo.** Sample data (842 clips, 1684 track/stac qpos CSVs, manual-label CSV) lives outside version control. If present locally, the startup autoload picks it up; otherwise the GUI opens empty and the user loads videos via the File menu.

Core features: multi-video session with lazy metadata loading, hotkey press-and-hold annotation, drag-to-resize edges AND drag-to-translate segment bodies, drag-up/down on a segment to change its behavior label, editable behavior name/hotkey table, two-track CSV import/export, snapshot-based undo/redo (Ctrl/Cmd-Z, Ctrl/Cmd-Shift-Z), per-clip auto-labeling + auto-directionality from qpos heuristics, conflict overlay for auto-vs-manual disagreement on the behavior track.

## Invariants (post-refactor)

- **Vocabulary is fixed at five**: `Immobile, Rear, Turn, Walk, Groom`. Users can still add more via the Behaviors menu, but the auto-pipelines only emit those five.
- **Priority** in `auto_labeler._classify_frame`: `Rear > Turn > Groom > Walk > Immobile`. Encoded by order of if-statements; changing order silently changes classifier output.
- **Turn requires stationary body**: `v_xy < walk_xy` is part of the Turn rule. A walking-while-turning frame is Walk, not Turn.
- **Direction is a separate track**. No `BehaviorSegment.direction` field anymore. Direction segments live in `self.videos[name]["direction_segments"]` and `TimelineWidget.direction_segments`.
- **CSV schema has a `Track` column** (values: `behavior | direction`). Old 6-column CSVs and the intermediate Direction-flag 7-column CSVs are still imported via rename + synthesis in `_normalize_import_rows`.

## Code Style Requirements

- **Type hints on everything.** Parameters and returns are annotated. Uses `typing.Optional`, `List`, `Dict`, `Tuple`, `Any`. Method signatures that return nothing use `-> None` explicitly.
- **Google-style docstrings on every public method**, including `Args:` and `Returns:` sections. Class docstrings include `Attributes:` and (for widgets) `Signals:` sections.
- **Attribute type annotations inside `__init__`** (e.g. `self.duration: float = 1.0`).
- **Black-compatible formatting** (trailing commas in multi-line function calls, double quotes).
- **Imports:** stdlib first (`sys`, `cv2`, `csv`, `os`, `random`, `typing`), then PyQt5 blocks grouped by submodule (`QtWidgets`, `QtCore`, `QtGui`).
- **No comments explaining what code does**; only short comments marking state-machine transitions or Qt signal-handling quirks (e.g. `# Temporarily disconnect signal to avoid triggering during update`).

## Development Workflow

- Default branch: `main`. History starts with the initial two-track import from the upstream fork.
- No Conventional Commits — subject lines describe the change in plain English.
- No CI config present.
- GUI changes must be **verified manually by running the app** — `conda activate behavior_gui && python annotation_GUI.py`. Tests do not catch rendering, cursor, or interaction regressions.

## Testing Rules

- Framework: **pytest** with `PyQt5.QtTest.QTest` for simulated key/mouse events.
- Tests live in `tests/`. `tests/conftest.py` adds the repo root to `sys.path` and sets `ANNOTATION_GUI_NO_AUTOLOAD=1` so the sample-video autoload is skipped during testing.
- Fixtures defined at module top of `tests/test_annotation_GUI.py`:
  - `qapp` (session-scoped `QApplication` singleton)
  - `temp_video` (generates a 10-second MP4 at 30 fps via `cv2.VideoWriter` into `tmp_path`)
  - `temp_csv`, `gui` (GUI with loaded test video, uses `unittest.mock.patch` on `QFileDialog`)
- Tests organized into class groups (`TestBehaviorSegment`, `TestTimelineWidget`, etc.) matching source classes.
- **Mock file dialogs and `QMessageBox`** in any test that triggers user-interaction paths — otherwise tests block.
- Run: `pytest tests/ -v`.

## Research Context

<!-- What is this project? What problem does it solve? What domain is it in? -->
<!-- PENDING: This is an engineering tool (annotation GUI), not a research experiment. Fill in if it supports a specific research question in the broader topo-vnl project. -->


## Research Question

<!-- What are you testing or building? Be specific — not "implement X" but "test whether X improves Y under condition Z." -->
<!-- PENDING -->


## Hypothesis

<!-- What do you expect to happen and why? This is what auditors verify against. -->
<!-- PENDING -->


## Method

<!-- Which paper(s), which equations, which specific algorithmic choices? Link to papers, cite section numbers. Claude can't infer these from context. -->
<!-- PENDING -->


## Domain-Specific Pitfalls

PyQt5 / OpenCV GUI pitfalls observed in this codebase:

- **Signal recursion on `QTableWidget`**: `itemChanged` fires when you programmatically set items. The codebase handles this by calling `self.behavior_table.itemChanged.disconnect(...)` before bulk updates, and `blockSignals(True)` before reverting a single cell. If you add a new edit path, follow the same pattern or it will recurse.
- **Active annotations leak across video switches**: segments with `end_time=None` are "in progress". Always call `_stop_all_active_annotations()` before `switch_video` / `import_csv` / anything that mutates `self.cap`, or you'll produce segments that extend to random playhead positions in the wrong video.
- **Deep-copy segments at video boundaries**: `self.timeline.segments` and `self.videos[name]["segments"]` must be different objects. Use `_deep_copy_segments` on handoff. Editing a segment drag on the timeline should not silently mutate stored segments of another video. The same rule applies to `self.timeline.auto_segments` vs `self.videos[name]["auto_segments"]`.
- **Class-level color map on `BehaviorSegment`**: `_color_map` and `_used_colors` are **class variables**, shared across all instances and persistent across GUI sessions within one process. Renaming a behavior requires updating `_color_map` in sync; tests that create behaviors with unique names can exhaust the palette.
- **Frame/time decoupling**: `start_time` and `start_frame` are independent fields. If you adjust `start_time` (e.g. by dragging), you must also recompute `start_frame = int(start_time * fps)`. `on_segment_modified` does this — don't skip it in new code paths. Body-drag translation updates both `start_time` and `end_time`; frames are recomputed in `on_segment_modified` on release.
- **OpenCV codec seekability**: `cv2.VideoCapture` may load a video but return `fps=0` or fail to seek mid-file for certain codecs. The README recommends `ffmpeg ... -c:v libx264 -preset superfast` for problematic inputs.
- **Hotkey repeat events**: `QKeyEvent.isAutoRepeat()` must be filtered in `keyPressEvent`, otherwise holding a key creates many zero-duration segments. There's a recent commit fixing this ("fix repeated hot key press editing issue") — don't regress it. `start_behavior_annotation` itself guards via `active_hotkeys.get(behavior_name, False)` check, which also protects `_push_undo()` from firing repeatedly on auto-repeat.
- **Undo snapshots must happen BEFORE mutation, not after.** Every mutation site calls `self._push_undo()` at the top (before any state change). `segment_drag_started` (emitted from `mousePressEvent` in `TimelineWidget`) is wired to `_push_undo` so drags capture pre-drag state. Never push undo from an `on_*_modified` handler — that snapshots the post-mutation state.
- **Direction-flag invariant.** Every `BehaviorSegment` has `direction ∈ {"straight", "left", "right"}`. `__init__` clamps invalid values to `"straight"`. Exported CSVs have a `Direction` column (new 7-column schema); legacy 6-column CSVs are auto-transformed via `convert_labels.transform_rows` on import.
- **Video list `!` prefix is cosmetic-only.** `switch_video` strips the `"! "` prefix from `item.text()` before looking up `self.videos`. If you add UI that reads from `video_list` items, do the same lstrip, or key off `self.videos[name]["has_conflict"]` directly.
- **Autoload in tests.** `AnnotatorGUI.__init__` calls `_autoload_sample_dataset()` by default, which scans `sample/videos/` for 842 clips. Tests set `ANNOTATION_GUI_NO_AUTOLOAD=1` at module import. Any new test that constructs `AnnotatorGUI` must either set that env var or be prepared for 842 video-list items.
- **qpos column semantics are inferred, not documented.** `auto_labeler.py` treats q0–q2 as root xyz, q3–q6 as a quaternion, and q7–q30 as upper-body joints. If the rig changes, re-verify — rules are defined in `docs/auto_labeling.md`.

## Project Conventions

- **Single-file app** — resist the urge to split `annotation_GUI.py` into modules. The existing tests import `AnnotatorGUI`, `BehaviorSegment`, `TimelineWidget` from this one module.
- **Legacy widgets kept for backward compatibility**: `self.behavior_list` is hidden (`setVisible(False)`) but still updated via `update_behavior_list()` — tests reference it. Don't delete.
- **Video data shape**: `self.videos[name] = {"path": str, "segments": List[BehaviorSegment], "duration": float, "fps": float}`. The dict keys are filenames (basename), not full paths.
- **Hotkeys stored as Qt key codes** (`Qt.Key_I`, etc.), not characters. Use `QKeySequence(key).toString()` to display. Letter keys are `Key_A`...`Key_Z`; digits are `Key_0`...`Key_9`.
- **CSV schema is fixed**: columns `Video, Behavior, Start_Time, End_Time, Start_Frame, End_Frame` in that order. Empty fields = ongoing annotation.

## Known Constraints

- **Python 3.10** (conda env `behavior_gui`).
- **Pinned versions**: `pyqt5==5.15.11`, `pyqt5-qt5==5.15.18`, `pyqt5-sip==12.17.2`, `opencv-python==4.12.0.88`, `numpy==2.2.6`, `pytest==9.0.2`.
- **Platform**: macOS, Windows 10+, Linux. macOS is the primary dev platform (user: macOS Darwin 23.6.0, conda env prefix is mambaforge).
- **Memory**: videos stay in RAM via `cv2.VideoCapture` handles — large videos (>1 GB) load slowly; many concurrent videos can OOM a 4 GB machine.
- **No GPU/accelerator requirements.**

## What "Correct" Means Here

<!-- How do you verify this project works? Not just "tests pass" — what domain-specific checks matter? -->

Until the user confirms otherwise, assume:

- **Tests pass**: `pytest test_annotation_GUI.py` green.
- **App launches and is usable manually**: `python annotation_GUI.py` → load a video → annotate with a hotkey → see a segment on the timeline → drag its edge → export CSV → re-import CSV → see identical segments.
- **No regressions in interactive behavior**: hotkey press-and-hold produces exactly one segment per press (no repeats from auto-repeat), dragging segment edges updates both `start_time`/`end_time` *and* `start_frame`/`end_frame`, switching videos preserves each video's annotations independently.
- **CSV round-trip is lossless** for complete annotations (all four time/frame fields populated).
