"""GSI-MLC-PA v6.3.4: OOF-guarded SAMG candidate.

This class intentionally reuses the v6.3.3 pipeline.  The only behavioral
change is that the legacy clean-margin candidate must also pass an OOF
positive-F1/negative-F1 safety check before it is accepted.
"""

from .gsi_v6_3_3 import GSIMLCPAv6_3_3Classifier


class GSIMLCPAv6_3_4Classifier(GSIMLCPAv6_3_3Classifier):
    """Thin, explicit version boundary for the safer SAMG2 policy."""

    def __init__(self, **kwargs):
        kwargs.pop("samg_strategy", None)
        super().__init__(samg_strategy="oof_guarded_margin", **kwargs)

