<!-- Copyright (C) 2026 YIQISOFT; SPDX-License-Identifier: Apache-2.0 -->

# Provisioned devices

YAML files in this directory are auto-loaded by the device SDK at startup.
The service ships with one singleton manager device (`nereid-inference.yaml`)
for service-level model inventory and diagnostics. It is not an inference job:
media jobs are normally provisioned through EdgeX core-metadata (by NereidEdge
or an operator), so no camera/model devices are shipped by default.

For development, drop a file such as `device.inference.example.yaml` here:

```yaml
deviceList:
  - name: example-inference-01
    profileName: nereid-inference-detection
    description: Example inference job (development only)
    labels:
      - nereid
      - inference
    protocols:
      Inference:
        SourceKind: jpeg
        SourceURL: "http://camera-host/snapshot.jpg"
        ModelID: example-model
        ModelVersion: "1.0.0"
        TargetDevice: CPU
        Endpoint: "edgex-ovms:9000"
        Trigger: periodic
        PeriodMs: "5000"
```

A camera registered with `device-onvif-camera` can also use that service's
authenticated `Snapshot` command without copying camera credentials into this
service:

```yaml
    protocols:
      Inference:
        SourceKind: onvif-snapshot
        SourceDevice: registered-onvif-camera
        ModelID: example-model
        ModelVersion: "1.0.0"
        TargetDevice: CPU
```
