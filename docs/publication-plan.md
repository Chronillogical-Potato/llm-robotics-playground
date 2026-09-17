# Publication plan

The initial release will contain four curated experiment packages. Each should let someone understand the setup, run its controller, inspect its checks, and compare a new attempt with the recorded example.

## Package contents

Keep the structure small and organize by task. Add model-specific attempts inside the relevant experiment when they exist; a new model does not need a separate repository.

```text
experiments/
  headphone-untangling/
  robo-spider/
  fibonacci-writing/
  dove-drawing/
```

Each package needs a README, scene or scene builder, required starting state, controller, renderer/viewer, dependencies, validation, and asset provenance. Existing filenames can remain where that makes the package easier to maintain. Shared code should be extracted only when multiple tasks actually need it.

Commit small configuration files, initial states, metrics, and preview images. Put large recordings and trajectory bundles in versioned GitHub Releases, with a manifest linking the files to the matching code revision and recording their checksums. A clean clone should remain useful without downloading every video.

Retain enough controller and state information to reproduce the selected attempt. Archive older scene revisions, tuning sweeps, intermediate exports, and scratch scripts outside the default package. Summarize useful failures and preserve the data supporting any published failure claim.

## Selected versions

### Headphone untangling

Package the two-region environment and its accepted opening sequence. The chosen presentation removes a short grasp-retry interval and ends before later cleanup. Include its edit manifest and the continuous source recording/trajectory as separate artifacts.

The demonstrated outcome is opening both local tangled regions. The selected endpoint has not passed the strict complete-release verifier; do not label it a fully verified untangling success. Preserve that verifier for future attempts.

Make the robot assets and validation helpers independent of the separate wiring-repair experiment. Replace machine-specific paths while preserving the connection between scene versions and saved states. Document any path-only scene conversion and retain the original source hash.

### Robo spider

Package the final six-legged, dual-arm table-transfer configuration, its controller, command interface, trajectory, and contact/placement checks. The chosen video is labeled 2x playback.

Publish the model-authored controller as the demonstrated baseline. The presence of a camera interface does not establish that the recorded run used vision or live model piloting.

### Fibonacci writing

Package the floating-base humanoid, friction-held marker, writing controller, stroke definitions, prepared grasp state, and contact-based ink renderer. Include the final presentation and its separately simulated ending.

Document that the marker starts already grasped, the writing follows supplied glyph paths, and the controller observes simulator state. Keep the distinction between writing a Fibonacci program and executing that program.

### Dove drawing

Package the physical-grasp version: a free pencil held through hand contact, with marks generated from pencil-tip contact. Its prepared grasp and traced stroke paths are part of the setup.

Keep the earlier fixed-grasp version outside the default package. Record the origin and distribution terms of the reference image and derived assets before including them. Document the video speed and any display-only thickening of the contact marks.

## Before adding an experiment

1. Resolve local imports, external assets, licenses, and absolute paths.
2. Define and document the reset state, controller inputs, observations, and success criteria.
3. Check loading, a short physics run, and replay in an isolated environment; run relevant existing validators.
4. Distinguish checks rerun on the public package from historical checks on the original recording. Record platform and dependency versions.
5. Add a concise account of model involvement and human guidance. Use unknown for unavailable model settings; do not reconstruct an exact original prompt from memory.
6. Review the exact files and release artifacts, including metadata, before publishing them.

Private conversations, local account details, unrelated project files, caches, temporary environments, and exhaustive development output are outside the publication scope.
