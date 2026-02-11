#!/bin/bash

CONFIG_PY="./config.py"

# Show current config
echo "=== Current Config ==="
python3 "$CONFIG_PY" --get_all
echo "====================="

# Stop services safely using supervisorctl
echo "[INFO] Stopping all breadcount services via Supervisor..."
# Stopping the services in one command is generally fastest
sudo supervisorctl stop breadcount-uvicorn breadcount-main breadcount-ros2 breadcount-spool-recorder breadcount-spool-processor 2>/dev/null

# Check status using supervisorctl
echo "[INFO] Checking service status..."

# Retrieve the full status report
SERVICE_STATUS=$(sudo supervisorctl status breadcount-ros2 breadcount-main breadcount-uvicorn breadcount-spool-recorder breadcount-spool-processor 2>/dev/null)

# Define the list of services to check
SERVICES=("breadcount-ros2" "breadcount-main" "breadcount-uvicorn" "breadcount-spool-recorder" "breadcount-spool-processor")

# Check status for each service
for service in "${SERVICES[@]}"; do
    # Look for the service name followed by "STOPPED"
    if echo "$SERVICE_STATUS" | grep -q "$service.*STOPPED"; then
        echo "[INFO] $service stopped successfully."
    elif echo "$SERVICE_STATUS" | grep -q "$service"; then
        # If it's not STOPPED, it might be FATAL or RUNNING (indicating a failure to stop)
        # Use grep to capture the actual status line for better reporting
        STATUS_LINE=$(echo "$SERVICE_STATUS" | grep "$service")
        echo "[WARN] $service is still active or failed to stop!"
        echo "       Status: $STATUS_LINE"
    else
        echo "[INFO] $service not configured (skipping)"
    fi
done