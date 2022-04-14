import sys
from importlib import import_module

__version__ = "0.0.1"
hadErrors = False

try:
  from rich.console import Console
  console = Console()
except ImportError as e:
  print("Module `Console` is not installed!")
  class MyConsole:
    def print(content):
      print(content)
  console = MyConsole
  hadErrors = True

try:
  from rich.markdown import Markdown
except ImportError as e:
  print("Module `Markdown` is not installed!")
  Markdown = str
  hadErrors = True

def printHelp():
  with open("./README.md", encoding = 'utf-8') as readme:
    console.print(Markdown(readme.read()))
  if (hadErrors):
    print("""
      --> You probably didn't follow the instructions above, <--
      --> so be sure to check them out                       <--
    """)

if (hadErrors or "--help" in sys.argv or "--h" in sys.argv):
  printHelp()
elif ("--version" in sys.argv or "--v" in sys.argv):
  print(f"Current version is {__version__}")
elif ("--server" in sys.argv or "--serve" in sys.argv):
  print("Starting the server...\n")
  serverModule = import_module("apps.SimplificationServer.main")
  app = getattr(serverModule, "startSimplificationServerApp")
  app()
else:
  print("Loading the transformer model...\n")
  transformerModule = import_module("apps.Transformer.main")
  app = getattr(transformerModule, "startTransformerApp")
  app()
