import logging

class SpecificMessageFilter(logging.Filter):
    def filter(self, record):
        #suppress message
        return not record.getMessage().startswith(
            "Warning: Some classes do not exist in the target. F1 scores for these classes will be cast to zeros."
        )