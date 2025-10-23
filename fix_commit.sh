#!/bin/bash

# Set the editor to sed for automated editing
export EDITOR="sed -i 's/E-commerce Analytics Platform for Vercel deployment/E-commerce Analytics Dashboard Platform/'"

# Use git rebase to edit the commit
git rebase -i fde8a70~1

echo "Commit message updated successfully!"
