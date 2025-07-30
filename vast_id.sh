#!/bin/bash

if [ -n "$VAST_CONTAINERLABEL" ]; then
    echo "$VAST_CONTAINERLABEL"
else
    echo ""
fi