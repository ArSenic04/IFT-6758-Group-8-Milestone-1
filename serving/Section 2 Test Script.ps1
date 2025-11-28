Logs Link: 127.0.0.1:5000/logs

Example Test Script:

## Before running test for flask app, note I used Flask version 2.3 for running app.py, make sure you have severed the connection to any
## older flask runs, and have deleted any flask.logs pycaches or models which have been used before from your Visual Studio code or whatever platform you are running this from. 

## Note this presumes you run the test on windows, UBUNTU will need CURL commands 

# In the first VS Code Terminal, run line below 
python -m waitress --listen=0.0.0.0:5000 serving.app:app

# Create a new terminal and execute the following commands 
# Test logs
# Outputs message indicating successful loading of default model logreg_distance
# First few commands may fail due to server not being fully connected yet
# However, API call should execute properly after 2 to 3 minutes in the worst-case scenario
Invoke-WebRequest -Uri "http://127.0.0.1:5000/logs" | Select-Object -ExpandProperty Content

# Test default distance model, we should expect a probability value returned 
Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" -Method POST -ContentType "application/json" -Body '{"distance_from_net":[20]}'

# Wrong field test, we should expect an error which indicates incorrect field inputted. 
Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" -Method POST -ContentType "application/json" -Body '{"angle_from_net":[30]}'

# Switch to angle model, we should expect a successful switch to new model message  
Invoke-RestMethod -Uri "http://127.0.0.1:5000/download_registry_model" -Method POST -ContentType "application/json" -Body '{"model":"logreg_angle","version":"latest"}'

# We should expect a probability value returned here 
Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" -Method POST -ContentType "application/json" -Body '{"angle_from_net":[30]}'

# Switch to distance+angle model, we should see a successful message for model switch returned
Invoke-RestMethod -Uri "http://127.0.0.1:5000/download_registry_model" -Method POST -ContentType "application/json" -Body '{"model":"logreg_distance_angle","version":"latest"}'

# Predicting goal based on distance and angle combined, we should expect a probability output here
Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" -Method POST -ContentType "application/json" -Body '{"distance_from_net":[20], "angle_from_net":[30]}'

# Missing JSON body, should expect an error message
Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" -Method POST

# Wrong format (string instead of dict/list), should expect an error message 
Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" -Method POST `
-ContentType "application/json" -Body '"hello"'

# Empty JSON Body, should expect an error message
Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" -Method POST `
-ContentType "application/json" -Body '{}'

# Empty model to download, should expect an error message
Invoke-RestMethod -Uri "http://127.0.0.1:5000/download_registry_model" -Method POST `
-ContentType "application/json" `
-Body '{"model":"not_a_real_model", "version":"v999"}'

# Call successfully extracts latest version of logreg_distance, should expect successful message 
Invoke-RestMethod -Uri "http://127.0.0.1:5000/download_registry_model" -Method POST `
-ContentType "application/json" `
-Body '{"model":"logreg_distance", "version":"latest"}'

# Should see log message 
Invoke-WebRequest -Uri "http://127.0.0.1:5000/logs" | Select-Object -ExpandProperty Content

# Expect to see 2 successful predictions derived from distance 
## we drop angle_from_net feature quietly as they are unnecessary for the input 
Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" `
-Method POST -ContentType "application/json" `
-Body '{"distance_from_net":[20, 35], "angle_from_net":[30, 40]}'
