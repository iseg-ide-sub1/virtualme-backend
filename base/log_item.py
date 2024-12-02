from __future__ import annotations

import enum


class EventType(enum.Enum):
    OpenTextDocument = "OpenTextDocument"
    CloseTextDocument = "CloseTextDocument"
    ChangeTextDocument = "ChangeTextDocument"
    CreateFile = "CreateFile"
    DeleteFile = "DeleteFile"
    SaveFile = "SaveFile"
    RenameFile = "RenameFile"
    MoveFile = "MoveFile"
    AddTextDocument = "AddTextDocument"
    DeleteTextDocument = "DeleteTextDocument"
    EditTextDocument = "EditTextDocument"
    RedoTextDocument = "RedoTextDocument"
    UndoTextDocument = "UndoTextDocument"
    SelectText = "SelectText"
    MouseHover = "MouseHover"
    MouseClick = "MouseJump"
    OpenTerminal = "OpenTerminal"
    CloseTerminal = "CloseTerminal"
    ChangeActiveTerminal = "ChangeActiveTerminal"
    ExecuteTerminalCommand = "ExecuteTerminalCommand"
    ExecuteMenuItem = "ExecuteMenuItem"


class ArtifactType(enum.Enum):
    File = "File"
    Module = "Module"
    Namespace = "Namespace"
    Package = "Package"
    Class = "Class"
    Method = "Method"
    Property = "Property"
    Field = "Field"
    Constructor = "Constructor"
    Enum = "Enum"
    Interface = "Interface"
    Function = "Function"
    Variable = "Variable"
    Constant = "Constant"
    String = "String"
    Number = "Number"
    Boolean = "Boolean"
    Array = "Array"
    Object = "Object"
    Key = "Key"
    Null = "Null"
    EnumMember = "EnumMember"
    Struct = "Struct"
    Event = "Event"
    Operator = "Operator"
    TypeParameter = "TypeParameter"
    Terminal = "Terminal"
    MenuItem = "MenuItem"
    Unknown = "Unknown"


class ContextType(enum.Enum):
    Add = "Add"
    Delete = "Delete"
    Edit = "Edit"
    Redo = "Redo"
    Undo = "Undo"
    Select = "Select"
    Hover = "Hover"
    Terminal = "Terminal"
    Unknown = "Unknown"


class Artifact:
    def __init__(self,
                 name: str,
                 artifact_type: ArtifactType,
                 reference: list[Artifact] = None
                 ):
        self.name = name  # 用名字包含层级信息->父->子->...->当前
        self.artifact_type = artifact_type
        self.reference = reference
        self.count = 1

    def add_count(self):
        self.count += 1

    def __str__(self):
        ret = f" {self.name} ({self.artifact_type})\n"
        if self.reference:
            ret += f"  reference: \n"
            for ref in self.reference:
                ret += f"   {ref}\n"
        return ret


class Context:
    def __init__(self,
                 context_type: ContextType,
                 content: tuple[str, str],
                 start: tuple[int, int],
                 end: tuple[int, int]):
        self.context_type = context_type
        self.content = content
        self.start = start
        self.end = end
        self.count = 1

    def add_count(self):
        self.count += 1

    def get_cmd(self):
        if self.context_type == ContextType.Terminal:
            return self.content[0]
        else:
            raise ValueError(f"Wrong context type: {self.context_type}, expected: Terminal")

    def __str__(self):
        ret = ''
        if self.context_type == ContextType.Add:
            ret += f" Add '{self.content[1]}' at line '{self.start[0]}'"
        elif self.context_type == ContextType.Delete:
            ret += f" Delete '{self.content[0]}' at line '{self.start[0]}'"
        elif self.context_type == ContextType.Edit:
            ret += f" Edit '{self.content[0]}' to '{self.content[1]}' at line '{self.start[0]}'"
        elif self.context_type == ContextType.Redo:
            ret += f" Redo '{self.content[1]}' at line '{self.start[0]}'"
        elif self.context_type == ContextType.Undo:
            ret += f" Undo '{self.content[0]}' at line '{self.start[0]}'"
        elif self.context_type == ContextType.Select:
            ret += f" Select '{self.content[0]}' at line '{self.start[0]}'"
        elif self.context_type == ContextType.Hover:
            ret += f" Hover '{self.content[0]}' at line '{self.start[0]}'"
        elif self.context_type == ContextType.Terminal:
            ret += f" Execute '{self.content[0]}' in terminal and get '{self.content[1]}'"
        else:
            ret += f" Unknown context type '{self.context_type}'"
        return ret


class LogItem:
    def __init__(self,
                 id: int,
                 timestamp: str,
                 event_type: EventType,
                 artifact: Artifact = None,
                 context: Context = None):
        self.id = id
        self.timestamp = timestamp
        self.event_type = event_type
        self.artifact = artifact
        self.context = context

    def __str__(self):
        ret = f"[{self.id}] ({self.event_type.value})\n"
        ret += f"timestamp: {self.timestamp}\n"
        if self.artifact:
            ret += f"artifact: \n{self.artifact}\n"
        if self.context:
            ret += f"context: \n{self.context}\n"
        return ret


class Log:
    def __init__(self,
                 artifact_history: set[Artifact],
                 cmd_history: set[Context],
                 log_items: [LogItem]):
        self.artifact_history = artifact_history
        self.cmd_history = cmd_history
        self.log_items = log_items

    def __str__(self):
        ret = f"=====Artifact_history:=====\n"
        for artifact in self.artifact_history:
            ret += f"{artifact.name} : {artifact.count}\n"
        ret += f"=====Cmd_history:=====\n"
        for cmd in self.cmd_history:
            ret += f"{cmd.get_cmd()} : {cmd.count}\n"
        ret += f"=====Log_items:=====\n"
        for log_item in self.log_items:
            ret += f"{log_item}\n"
        return ret
