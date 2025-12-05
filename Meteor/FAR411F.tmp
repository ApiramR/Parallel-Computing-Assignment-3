@echo off
for /l %%x in (1, 1, 10) do (
    gen.exe > data.txt
    sol.exe < data.txt > output.txt
    cuda.exe < data.txt > output2.txt
    fc output.txt output2.txt > diagnostics || exit /b
    echo %%x
)
echo all tests passed