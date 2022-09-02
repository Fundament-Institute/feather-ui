# Append local include directory to the include-search-path on windows. YOU MUST BUILD USING THE -e OPTION: ./scopes.exe -e blah.sc
'bind-symbols __env
    include-search-path = (cons str"./include" __env.include-search-path)