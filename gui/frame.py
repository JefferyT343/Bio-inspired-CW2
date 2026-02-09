import wx
import importlib.util
import sys
import time
import threading

from copy import deepcopy
from pathlib import Path
# from core.simulation import Simulation
from gui.utils import AppIdentifiers as ID
from gui.canvas import Canvas
from core.world.world import World
from core.log import LogWindow

class Frame(wx.Frame):
    def __init__(
        self,
        parent: wx.Frame,
        title: str,
        position: wx.Point = wx.Point(50, 50),
        size: wx.Size = wx.Size(808, 681),
        simulation = None
    ):
        super().__init__(parent, -1, title, position, size, wx.DEFAULT_FRAME_STYLE)
        self.simulation = simulation
        self.simulation_names = []
        self.simulation_class = []
        self.current_simulation = None
        self.world_canvas: Canvas = None
        self.current_thread: threading.Thread = None
        self.menu_bar = None
        self.demo_bar = None
        self.status_bar = self.CreateStatusBar(2)
        self.log_window: LogWindow = None
        self.current_simulation_id: int = -1
        self.fps = 60
        self.started, self.paused = False, False
        self.initial_simulation_copy = None
        
        self.create_menu_bar()
        #self.create_log_window()
        self.SetMenuBar(self.menu_bar)
        self.status_bar.SetStatusText("Ready", 0)
    
    def load_demos(self) -> None:
        demo_directory = Path("./demos/")
        demos = [ f for f in demo_directory.iterdir() if f.suffix == ".py" ]
        for demo in demos:
            spec = importlib.util.spec_from_file_location(demo.stem, demo)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        
            if hasattr(module, "IS_DEMO") and module.IS_DEMO:
                sys.modules[demo.stem] = module
                name = getattr(module, "DEMO_NAME") # Demo Title
                class_name = getattr(module, "CLASS_NAME") # Name of Simulation Class
                sim_class = getattr(module, class_name)
                self.simulation_names.append(name)
                self.simulation_class.append(sim_class)
    
    def create_menu_bar(self) -> None:
        self.menu_bar = wx.MenuBar()
        
        file_menu = wx.Menu()
        # file_menu.Append(ID.FILE_SAVE, "&Save")
        # file_menu.Append(ID.FILE_LOAD, "&Load")
        file_menu.Append(ID.ABOUT, "&About")
        self.menu_bar.Append(file_menu, "&File")
        
        self.load_demos()
        demo_menu = wx.Menu()
        for i, demo in enumerate(self.simulation_names):
            demo_menu.Append(getattr(ID, f"START_SIM_{i}"), f"&{demo}")
        self.menu_bar.Append(demo_menu, "&Demo")
        
        action_menu = wx.Menu()
        if self.simulation is not None:
            action_menu.Append(ID.START_MY_SIM, "&Start")
        action_menu.Append(ID.SIMULATION_PAUSE, "&Toggle Pause")
        action_menu.Append(ID.SIMULATION_FAST, "&Toggle High Speed")
        action_menu.Append(ID.SIMULATION_RESET, "&Reset")
        self.menu_bar.Append(action_menu, "&Action")
        
        # Event Bindings
        self.Bind(wx.EVT_SIZE, self.on_resize)
        self.Bind(wx.EVT_CLOSE, self.on_exit)
        
        # self.Bind(wx.EVT_MENU, self.on_save, id=ID.FILE_SAVE)
        # self.Bind(wx.EVT_MENU, self.on_load, id=ID.FILE_LOAD)
        self.Bind(wx.EVT_MENU, self.on_about, id=ID.ABOUT)
        
        for item in demo_menu.GetMenuItems():
            self.Bind(wx.EVT_MENU, self.on_start_demo, item)
        
        if self.simulation is not None:
            self.Bind(wx.EVT_MENU, self.on_start_simulation, id=ID.START_MY_SIM)
        self.Bind(wx.EVT_MENU, self.on_pause, id=ID.SIMULATION_PAUSE)
        self.Bind(wx.EVT_MENU, self.on_fast, id=ID.SIMULATION_FAST)
        self.Bind(wx.EVT_MENU, self.on_reset, id=ID.SIMULATION_RESET)
    
    def create_log_window(self) -> None:
        position, size = self.GetPosition(), self.GetSize()
        self.log_window = LogWindow(self)
        self.log_window.SetPosition(wx.Point((position.x + size.GetWidth() + 10), position.y))
        self.log_window.SetSize(400, size.GetHeight())
        self.log_window.Show()
        
    def on_exit(self, event):
        if self.current_thread is not None:
            self.kill_simulation()
        if self.log_window is not None:
            self.log_window.Destroy()
        self.Destroy()
    
    def on_resize(self, event):
        if self.world_canvas is not None:
            if not self.paused:
                self.on_pause
            self.world_canvas.on_size(event)
        self.Refresh()
    
    def on_start_simulation(self, event):
        self.start_simulation(self.simulation)
    
    def on_start_demo(self, event):
        gui_name = self.menu_bar.FindItemById(event.GetId()).GetItemLabel()
        simulation = self.simulation_class[self.simulation_names.index(gui_name[1:])]()
        self.start_simulation(simulation)
        
    def on_pause(self, event) -> None:
        if self.current_simulation is None:
            return
        
        if not self.paused:
            self.pause_event.clear()
            self.paused = True
        else:
            self.pause_event.set()
            self.paused = False
    
    def on_fast(self, event) -> None:
        if self.render_simulation.is_set():
            self.render_simulation.clear()
        else:
            self.render_simulation.set()
    
    def on_about(self, event):
        wx.MessageBox("""Bioinspired Evolutionary Agent Simulation Toolkit\n
                      PyBEAST++ Version: 1.0.0\n
                      PyBEAST developed by University of Leeds
                      PyBEAST++ created by James Borgars from PyBEAST
                      """, "PyBEAST++", wx.ICON_INFORMATION)
    
    def on_reset(self, event) -> None:
        if self.initial_simulation_copy is not None:
            self.start_simulation(self.initial_simulation_copy)
    
    def start_simulation(self, simulation) -> None:
        self.initial_simulation_copy = deepcopy(simulation)
        
        if self.current_thread is not None:
            self.kill_simulation()
        
        self.current_simulation = simulation
        if not self.current_simulation.loaded:
            self.current_simulation.initialise()
        self.create_world_canvas(self.current_simulation.world)
        #self.current_simulation.log.addHandler(self.log_window.handler)
        #self.log_window.log_ctrl.Clear()
        
        self.pause_event = threading.Event()
        self.render_simulation = threading.Event()
        self.kill_thread = threading.Event()
        self.current_thread = threading.Thread(
            target=self.run_simulation,
            args=(self.pause_event, self.render_simulation, self.kill_thread)
        )
        self.current_thread.daemon = True
        self.current_thread.start()
        
    def kill_simulation(self):
        self.kill_thread.set()
        self.current_thread.join()
        time.sleep(0.25)
        self.destroy_world_canvas()
        
    def create_world_canvas(self, world: World):
        self.world_canvas = Canvas(self, self.GetClientSize(), world)
        event = wx.SizeEvent(self.GetClientSize())
        self.world_canvas.on_size(event)
    
    def destroy_world_canvas(self):
        self.world_canvas.Destroy()
        self.world_canvas = None
        self.current_simulation = None
        self.current_thread = None
        
    def run_simulation(
        self,
        pause_event: threading.Event,
        render_simulation: threading.Event,
        kill_thread: threading.Event
    ) -> None:
        time.sleep(0.2)
        if not self.current_simulation.loaded:
            self.current_simulation.begin_simulation()
        else:
            self.current_simulation.resume_simulation()
        pause_event.set()
        render_simulation.set()
        
        complete = False
        while not complete:
            pause_event.wait()
            if kill_thread.is_set():
                break
            start_time = time.time()
            complete = self.current_simulation.update()
            if render_simulation.is_set():
                wx.CallAfter(self.world_canvas.display)
                sleep_for = max(0.01, (1.0 / self.fps) - (time.time() - start_time))
                time.sleep(sleep_for)
            if complete:
                break
