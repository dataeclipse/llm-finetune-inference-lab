class LabError(Exception):
    pass


class TrainingError(LabError):
    pass


class ServingError(LabError):
    pass


class ExportError(LabError):
    pass
