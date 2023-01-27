# Based on `FindCaffe.cmake`
"""
unset(NDI_FOUND)
unset(NDI_INCLUDE_DIRS)
unset(NDI_LIB)

find_path(NDI_INCLUDE_DIRS NAMES
  NDI.h
  HINTS
  C:/Program Files/NDI/NDI 5 SDK/Include
  )

find_library(NDI_LIB NAMES NDI
    HINTS    
    C:/Program Files/NDI/NDI 5 SDK/Lib/x64)

if (NDI_INCLUDE_DIRS AND NDI_LIB)
  set(NDI_FOUND 1)
endif (NDI_INCLUDE_DIRS AND NDI_LIB)
"""