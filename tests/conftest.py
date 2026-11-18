"""twin.rt reads TEAM_ID and RAWTREE_API_KEY at import time; the tests never touch the network."""
import os

os.environ.setdefault("TEAM_ID", "team_00")
os.environ.setdefault("RAWTREE_API_KEY", "test")
