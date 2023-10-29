# Notes

## Streamlit

### My apps

[DSM2023](https://dsm2023.streamlit.app/)
[CPM](https://sed-cpm.streamlit.app/)

### Extensions

[Select Image](https://image-select.streamlit.app/)

### State

[Add statefulness to apps](https://docs.streamlit.io/library/advanced-features/session-state)

[Demo Mode: Using State in Streamlit](https://dev.to/alexeversmeyer/demo-mode-using-state-in-streamlit-1482)

If a widget disappears, its key will be removed from session state automatically.
There is a hack to get around this, which is to explicitly set the session state.
If you add this at the bottom of your “change window” calback, then the problem goes away:

´´´python
    for k, v in st.session_state.items():
        st.session_state[k] = v
´´´

### Cookies

[Extra components](https://github.com/Mohamed-512/Extra-Streamlit-Components)

## Visulaization

### Vega

[Example Gallery](https://vega.github.io/vega-lite/examples/)

### Altair

[Example Gallery](https://altair-viz.github.io/gallery/index.html)

[Simple heatmap](https://altair.streamlit.app/Simple_Heatmap)

### ECharts

[Streamlit ECharts Demos](https://echarts.streamlit.app/)

[option docs](https://echarts.apache.org/en/option.html#title)

### RaGraph

[Plotting](https://ragraph.ratio-case.nl/how-to-guides/plotting/)

## Firestore

[Firestoe console for dsm2023isw project](https://console.firebase.google.com/project/dsm2023isw/firestore/data)

[January 2021 - Streamlit firestore blog](https://blog.streamlit.io/streamlit-firestore/)
[April 2021 - Streamlit firestore blog continuation](https://blog.streamlit.io/streamlit-firestore-continued/)

[April 2022 -Firestore × Streamlit: a Powerful Combo for a Perfect Web App](https://levelup.gitconnected.com/4-easy-steps-to-set-up-a-firestore-database-for-your-streamlit-apps-825c5de5b5bc)

## Pandas

[10 minutes to pandas](https://pandas.pydata.org/docs/user_guide/10min.html)
[25 Nooby Pandas Coding Mistakes You Should NEVER make.](https://www.youtube.com/watch?v=_gaAoJBMJ_Q)


## Other

[Barfi: A visual Flow Based Programming library](https://barfi.readthedocs.io/en/latest/index.html)
