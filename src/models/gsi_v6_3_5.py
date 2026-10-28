"""GSI-MLC-PA v6.3.5: reject only dependent labels (DL)."""

from .gsi_v6_3_4 import GSIMLCPAv6_3_4Classifier


class GSIMLCPAv6_3_5Classifier(GSIMLCPAv6_3_4Classifier):
    """SAMG2 model with complete decisions for IL and abstention only on DL."""

    def __init__(self, **kwargs):
        kwargs.pop("rejection_scope", None)
        super().__init__(rejection_scope="dl_only", **kwargs)
