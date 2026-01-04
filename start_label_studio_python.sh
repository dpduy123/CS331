#!/bin/bash
# Start Label Studio with local file serving enabled

# Set environment variables for local file serving
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="/Users/nguyenloan/Desktop/CS331/CoffeeBeanDataset"

echo "🚀 Starting Label Studio with local file serving..."
echo "📁 Document root: /Users/nguyenloan/Desktop/CS331/CoffeeBeanDataset"
echo "🌐 URL: http://localhost:8080"
echo ""
echo "✅ Local file serving: ENABLED"
echo ""
echo "Press Ctrl+C to stop Label Studio"
echo ""

# Try to find label-studio executable
LABEL_STUDIO_PATH="/Library/Frameworks/Python.framework/Versions/3.12/bin/label-studio"

if [ -f "$LABEL_STUDIO_PATH" ]; then
    # Use full path if found with explicit flags
    "$LABEL_STUDIO_PATH" start \
      --host 0.0.0.0 \
      --port 8080 \
      --data-dir "$HOME/Library/Application Support/label-studio" \
      --log-level DEBUG
else
    # Fallback: try from PATH
    label-studio start \
      --host 0.0.0.0 \
      --port 8080 \
      --data-dir "$HOME/Library/Application Support/label-studio" \
      --log-level DEBUG
fi
