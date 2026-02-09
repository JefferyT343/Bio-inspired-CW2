import wx

from wx.glcanvas import GLCanvas, GLContext
from OpenGL.GL import *
from core.world.world import World

class Canvas(GLCanvas):
    def __init__(self, parent: wx.Window, size: wx.Size, world: World):
        super().__init__(
            parent,
            id=wx.ID_ANY,
            pos=wx.DefaultPosition,
            size=size,
            style=wx.WANTS_CHARS,
            name="GLCanvas"
        )
        
        self.context: GLContext = None
        self.initialised: bool = False
        self.world: World = world
        self._initialise_gl()
        self.Bind(wx.EVT_PAINT, self.on_paint)
    
    def _initialise_gl(self) -> None:
        self.context = GLContext(self)
        self.SetCurrent(self.context)
        
        # NOTE: does this work?
        size = self.GetSize()
        
        glViewport(0, 0, size.width, size.height)
        glFinish()
        self.world._initialise_gl()
        glFinish()
        self.initialised = True
    
    def display(self) -> None:
        self.SetCurrent(self.context)
        self.world.display()
        glFinish()
        self.SwapBuffers()
    
    def clean(self) -> None:
        return()
#        self.SetCurrent(self.context)
#        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    def on_paint(self, event):
        if not self.initialised:
            self._initialise_gl()
        wx.PaintDC(self)
        self.display()
    
    def on_size(self, event):
        width, height = event.GetSize()
        self.SetSize(width, height)
        self.SetCurrent(self.context)
        glViewport(0, 0, width, height)
        glFinish()
