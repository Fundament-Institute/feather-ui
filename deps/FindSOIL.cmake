set(SOIL_FIND_ARGS
  HINTS
    ENV SOIL_DIR
  PATHS
    ENV GTKMM_BASEPATH
    [HKEY_CURRENT_USER\\SOFTWARE\\gtkmm\\2.4;Path]
    [HKEY_LOCAL_MACHINE\\SOFTWARE\\gtkmm\\2.4;Path]
)

find_path(
  SOIL_INCLUDE_DIRS
  SOIL.h
  ${SOIL_FIND_ARGS}
  PATH_SUFFIXES
    include/SOIL
    include
    SOIL
)

if(NOT SOIL_LIBRARY)
  find_library(SOIL_LIBRARY_RELEASE
    NAMES
      SOIL
      libSOIL
    ${SOIL_FIND_ARGS}
    PATH_SUFFIXES
      lib
  )
  find_library(SOIL_LIBRARY_DEBUG
    NAMES
      SOILd
      libSOILd
    ${SOIL_FIND_ARGS}
    PATH_SUFFIXES
      lib
  )
  include(SelectLibraryConfigurations)
  select_library_configurations(SOIL)
else()
  # on Windows, ensure paths are in canonical format (forward slashes):
  file(TO_CMAKE_PATH "${SOIL_LIBRARY}" SOIL_LIBRARY)
endif()

unset(SOIL_FIND_ARGS)

# set the user variables
set(SOIL_LIBRARIES "${SOIL_LIBRARY}")
set(SOIL_VERSION_STRING "1.0.0")

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(
  SOIL
  REQUIRED_VARS
    SOIL_LIBRARY
    SOIL_INCLUDE_DIRS
  VERSION_VAR
    SOIL_VERSION_STRING
)

if(SOIL_FOUND)
  if(NOT TARGET SOIL::SOIL)
    add_library(SOIL::SOIL UNKNOWN IMPORTED)
    set_target_properties(SOIL::SOIL PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${SOIL_INCLUDE_DIRS}")

    if(SOIL_LIBRARY_RELEASE)
      set_property(TARGET SOIL::SOIL APPEND PROPERTY
        IMPORTED_CONFIGURATIONS RELEASE)
      set_target_properties(SOIL::SOIL PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
        IMPORTED_LOCATION_RELEASE "${SOIL_LIBRARY_RELEASE}")
    endif()

    if(SOIL_LIBRARY_DEBUG)
      set_property(TARGET SOIL::SOIL APPEND PROPERTY
        IMPORTED_CONFIGURATIONS DEBUG)
      set_target_properties(SOIL::SOIL PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "C"
        IMPORTED_LOCATION_DEBUG "${SOIL_LIBRARY_DEBUG}")
    endif()

    if(NOT SOIL_LIBRARY_RELEASE AND NOT SOIL_LIBRARY_DEBUG)
      set_target_properties(SOIL::SOIL PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES "C"
        IMPORTED_LOCATION "${SOIL_LIBRARY}")
    endif()
  endif()
endif()
