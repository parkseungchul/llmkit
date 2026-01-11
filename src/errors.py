class LlmKitError(Exception):
    pass

class ConfigError(LlmKitError):
    pass

class AdapterError(LlmKitError):
    pass

class ValidationError(LlmKitError):
    pass
