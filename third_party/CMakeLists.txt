set(THIRD_PARTY_MODULES gnuplot-cpp)
                        

set(LINK_LIBS "")
foreach(TMP_LIB ${THIRD_PARTY_MODULES})
  add_subdirectory(${TMP_LIB})
  set(LINK_LIBS ${LINK_LIBS} mats_${TMP_LIB})
endforeach(TMP_LIB)

set(MATS_LINK_LIBS ${MATS_LINK_LIBS} PARENT_SCOPE)
