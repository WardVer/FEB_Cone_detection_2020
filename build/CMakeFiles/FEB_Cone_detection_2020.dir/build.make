# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ward/yolow/FEB_Cone_detection_2020

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ward/yolow/FEB_Cone_detection_2020/build

# Include any dependencies generated for this target.
include CMakeFiles/FEB_Cone_detection_2020.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/FEB_Cone_detection_2020.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/FEB_Cone_detection_2020.dir/flags.make

CMakeFiles/FEB_Cone_detection_2020.dir/src/detect_cone.cpp.o: CMakeFiles/FEB_Cone_detection_2020.dir/flags.make
CMakeFiles/FEB_Cone_detection_2020.dir/src/detect_cone.cpp.o: ../src/detect_cone.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ward/yolow/FEB_Cone_detection_2020/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/FEB_Cone_detection_2020.dir/src/detect_cone.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/FEB_Cone_detection_2020.dir/src/detect_cone.cpp.o -c /home/ward/yolow/FEB_Cone_detection_2020/src/detect_cone.cpp

CMakeFiles/FEB_Cone_detection_2020.dir/src/detect_cone.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/FEB_Cone_detection_2020.dir/src/detect_cone.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ward/yolow/FEB_Cone_detection_2020/src/detect_cone.cpp > CMakeFiles/FEB_Cone_detection_2020.dir/src/detect_cone.cpp.i

CMakeFiles/FEB_Cone_detection_2020.dir/src/detect_cone.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/FEB_Cone_detection_2020.dir/src/detect_cone.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ward/yolow/FEB_Cone_detection_2020/src/detect_cone.cpp -o CMakeFiles/FEB_Cone_detection_2020.dir/src/detect_cone.cpp.s

CMakeFiles/FEB_Cone_detection_2020.dir/src/detect_cone.cpp.o.requires:

.PHONY : CMakeFiles/FEB_Cone_detection_2020.dir/src/detect_cone.cpp.o.requires

CMakeFiles/FEB_Cone_detection_2020.dir/src/detect_cone.cpp.o.provides: CMakeFiles/FEB_Cone_detection_2020.dir/src/detect_cone.cpp.o.requires
	$(MAKE) -f CMakeFiles/FEB_Cone_detection_2020.dir/build.make CMakeFiles/FEB_Cone_detection_2020.dir/src/detect_cone.cpp.o.provides.build
.PHONY : CMakeFiles/FEB_Cone_detection_2020.dir/src/detect_cone.cpp.o.provides

CMakeFiles/FEB_Cone_detection_2020.dir/src/detect_cone.cpp.o.provides.build: CMakeFiles/FEB_Cone_detection_2020.dir/src/detect_cone.cpp.o


# Object files for target FEB_Cone_detection_2020
FEB_Cone_detection_2020_OBJECTS = \
"CMakeFiles/FEB_Cone_detection_2020.dir/src/detect_cone.cpp.o"

# External object files for target FEB_Cone_detection_2020
FEB_Cone_detection_2020_EXTERNAL_OBJECTS =

FEB_Cone_detection_2020: CMakeFiles/FEB_Cone_detection_2020.dir/src/detect_cone.cpp.o
FEB_Cone_detection_2020: CMakeFiles/FEB_Cone_detection_2020.dir/build.make
FEB_Cone_detection_2020: CMakeFiles/FEB_Cone_detection_2020.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ward/yolow/FEB_Cone_detection_2020/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable FEB_Cone_detection_2020"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/FEB_Cone_detection_2020.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/FEB_Cone_detection_2020.dir/build: FEB_Cone_detection_2020

.PHONY : CMakeFiles/FEB_Cone_detection_2020.dir/build

CMakeFiles/FEB_Cone_detection_2020.dir/requires: CMakeFiles/FEB_Cone_detection_2020.dir/src/detect_cone.cpp.o.requires

.PHONY : CMakeFiles/FEB_Cone_detection_2020.dir/requires

CMakeFiles/FEB_Cone_detection_2020.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/FEB_Cone_detection_2020.dir/cmake_clean.cmake
.PHONY : CMakeFiles/FEB_Cone_detection_2020.dir/clean

CMakeFiles/FEB_Cone_detection_2020.dir/depend:
	cd /home/ward/yolow/FEB_Cone_detection_2020/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ward/yolow/FEB_Cone_detection_2020 /home/ward/yolow/FEB_Cone_detection_2020 /home/ward/yolow/FEB_Cone_detection_2020/build /home/ward/yolow/FEB_Cone_detection_2020/build /home/ward/yolow/FEB_Cone_detection_2020/build/CMakeFiles/FEB_Cone_detection_2020.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/FEB_Cone_detection_2020.dir/depend

