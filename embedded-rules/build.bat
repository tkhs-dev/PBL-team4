cd /d %~dp0
echo To build embedded-rules, you need to install gcc and add it to the PATH.
go build -buildmode=c-shared  -o build/rules.dll main.go