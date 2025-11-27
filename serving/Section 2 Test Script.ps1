Logs Link: 127.0.0.1:5000/logs

Example Test Script:

## Note this presumes you run the test on windows 

# In the first VS Code Terminal, run line below 
python -m waitress --listen=0.0.0.0:5000 serving.app:app

# Create a new terminal and execute the following commands 
# Test logs
Invoke-RestMethod -Uri "http://127.0.0.1:5000/logs" -Method GET

# Test default distance model
Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" -Method POST -ContentType "application/json" -Body '{"distance_from_net":[20]}'

# Wrong field test
Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" -Method POST -ContentType "application/json" -Body '{"angle_from_net":[30]}'

# Switch to angle model
Invoke-RestMethod -Uri "http://127.0.0.1:5000/download_registry_model" -Method POST -ContentType "application/json" -Body '{"model":"logreg_angle","version":"latest"}'
Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" -Method POST -ContentType "application/json" -Body '{"angle_from_net":[30]}'

# Switch to distance+angle model
Invoke-RestMethod -Uri "http://127.0.0.1:5000/download_registry_model" -Method POST -ContentType "application/json" -Body '{"model":"logreg_distance_angle","version":"latest"}'
Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" -Method POST -ContentType "application/json" -Body '{"distance_from_net":[20], "angle_from_net":[30]}'

# Missing JSON body
Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" -Method POST

# Wrong format (string instead of dict/list)
Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" -Method POST `
-ContentType "application/json" -Body '"hello"'

Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" -Method POST `
-ContentType "application/json" -Body '{}'

Invoke-RestMethod -Uri "http://127.0.0.1:5000/download_registry_model" -Method POST `
-ContentType "application/json" `
-Body '{"model":"not_a_real_model", "version":"v999"}'

Invoke-RestMethod -Uri "http://127.0.0.1:5000/download_registry_model" -Method POST `
-ContentType "application/json" `
-Body '{"model":"logreg_distance", "version":"latest"}'

Invoke-RestMethod -Uri "http://127.0.0.1:5000/logs"

Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" `
-Method POST -ContentType "application/json" `
-Body '{"distance_from_net":[20, 35], "angle_from_net":[30, 40]}'
