## You can directly run this file via python test_gclient in VS Code or Python terminal
## test_client will be directly imported 

from game_client import GameClient
import time

# Pick any REAL game_id from past seasons — this one always works
TEST_GAME_ID = "2023020003"  

gclient = GameClient(TEST_GAME_ID)

print("First Extraction run (should fetch many events) ")
df1 = gclient.extract()
print(df1.head())
print("Rows fetched:", len(df1))

# Extraction again — should fetch 0 new events
print("\n Second Extraction run (should fetch ZERO, as the game has been seen previously)")
time.sleep(2)  # not required, but simulates live delay
df2 = gclient.extract()
print(df2.head())
print("Rows fetched:", len(df2))

# Extraction for the third time, should be zero again.
print("\n Third Extraction run ")
time.sleep(3)
df3 = gclient.extract()
print("Rows fetched:", len(df3))
