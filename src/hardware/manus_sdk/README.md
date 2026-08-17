# Manus Core 3.1.1 runtime

`client.py` is derived from the working EgoTac4D Manus adapter at commit
`21fa87b2940e01809a04e76ed80bea5fb95ed993`. This repository retains only the
Integrated SDK hand-skeleton, glove-status, settings, and calibration paths.
EgoTac4D's tracker feed, tracker read-back, and tracker-to-wrist alignment APIs
are removed. The local command socket remains available for MANUS settings and
official glove calibration.

Place the Linux Integrated SDK library here:

```text
lib/libManusSDK_Integrated.so
```

The SDK binary is intentionally ignored by Git. Its use is governed by
`MANUS_SDK_LICENSE.txt`.
