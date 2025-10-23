#!/bin/bash

# Create a script to handle the rebase
cat > /tmp/rebase_editor.sh << 'EOF'
#!/bin/bash
# This script will be used as the editor for git rebase
# It will automatically change the commit messages
sed -i 's/E-commerce Analytics Platform for Vercel deployment/E-commerce Analytics Dashboard Platform/g' "$1"
sed -i 's/Update E-commerce Analytics Platform for Vercel deployment/Update E-commerce Analytics Dashboard Platform for Vercel deployment/g' "$1"
EOF

chmod +x /tmp/rebase_editor.sh

# Set the editor
export EDITOR="/tmp/rebase_editor.sh"

# Perform the rebase
git rebase -i 07844c7

echo "Commit messages updated successfully!"
