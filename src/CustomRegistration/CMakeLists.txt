#-----------------------------------------------------------------------------
set(MODULE_NAME CustomRegistration)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
    ${MODULE_NAME}.py
)

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
    NAME ${MODULE_NAME}
    SCRIPTS ${MODULE_PYTHON_SCRIPTS}
)
