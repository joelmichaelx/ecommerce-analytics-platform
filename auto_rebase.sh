#!/bin/bash

# Create a script that will be used as the editor
cat > /tmp/rebase_script.sh << 'EOF'
#!/bin/bash
# This script will be used as the editor for git rebase
# It will automatically change the commit message
sed -i 's/pick fde8a70/pick fde8a70\nsquash d7b4e69/' "$1"
sed -i 's/E-commerce Analytics Platform for Vercel deployment/E-commerce Analytics Dashboard Platform/' "$1"
EOF

chmod +x /tmp/rebase_script.sh

# Set the editor to our script
export EDITOR="/tmp/rebase_script.sh"

# Perform the rebase
git rebase -i HEAD~2

echo "Rebase completed!"
