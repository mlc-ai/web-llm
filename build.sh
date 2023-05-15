MLC_LLM_HOME_SET="${MLC_LLM_HOME:-}"

if [ -z ${MLC_LLM_HOME_SET} ]; then
    export MLC_LLM_HOME="${MLC_LLM_HOME:-mlc-llm}"
fi


cd ${MLC_LLM_HOME}
python build.py --target webgpu ${@}
cd -