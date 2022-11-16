#!/bin/bash
CURRENT=$(cd $(dirname $0);pwd)
cd $CURRENT
cd ../terrains/x
pwd
rm -f generate*
cd ../y
rm -f generate*
