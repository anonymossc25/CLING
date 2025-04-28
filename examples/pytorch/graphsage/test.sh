#!/bin/bash

# Variable containing a value
variable="123ab"

# Regular expression to match floating-point numbers
if [[ $variable =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "$variable is a floating-point number."
else
    echo "$variable is not a floating-point number."
fi
