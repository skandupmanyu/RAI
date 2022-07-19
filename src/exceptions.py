class RAIError(Exception):
    """Base exception class for all custom errors"""
    pass

class DatasetNotFoundError(RAIError):
    """The dataset was not found on disk"""
    pass

class ModelUnknownError(RAIError):
    """The requested model does not exist in the model registry"""
    pass

class DirectoryNotFoundError(RAIError):
    """The dataset was not found on disk"""
    pass