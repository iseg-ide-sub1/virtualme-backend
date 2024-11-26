

class Artifact:
    def __init__(self, name, artifact_type, line: int, hierarchy: list):
        self.name = name
        self.artifact_type = artifact_type
        self.line = line
        self.hierarchy = hierarchy

    def __str__(self):
        ret = ""
        for artifact in self.hierarchy:
            ret += artifact.name + "->"
        ret += ": " + str(self.line)
        
        return ret

