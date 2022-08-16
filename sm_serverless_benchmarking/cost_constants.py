
INFERENCE_COST = {1024:0.0000200 / 1000, 
                  2048:0.0000400 / 1000, 
                  3072:0.0000600 / 1000, 
                  4096:0.0000800 / 1000, 
                  5120:0.0001000 / 1000, 
                  6144:0.0001200 / 1000}

PROCESSING_COST = 0.016 / (1024**3)

INSTANCE_MAPPING = {1024: "ml.t2.medium", 
                    2048: "ml.t2.medium", 
                    3072: "ml.c5.large", 
                    4096: "ml.c5.large", 
                    5120: "ml.c5.large", 
                    6144: "ml.c5.xlarge"}

MONTHLY_INSTANCE_COST = {"ml.t2.medium": 0.056*24*30, 
                         "ml.c5.large": 0.102*24*30, 
                         "ml.c5.xlarge": 0.204*24*30}