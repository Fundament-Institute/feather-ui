@echo off
hugo
call git checkout master
RMDIR /S /Q docs
move /-y public docs