from streamlit.testing.v1 import AppTest
at = AppTest.from_file("app.py", default_timeout=120).run()
assert not at.exception, at.exception
print("default ok | metrics:", [m.value for m in at.metric][:5], "| tabs:", len(at.tabs))
# interact: tighten filters, toggle checkboxes
at.sidebar.slider[0].set_value((2010, 2023)).run(); assert not at.exception, at.exception
at.sidebar.checkbox[0].check().run(); assert not at.exception, at.exception
print("HZ-only ok | planets:", at.metric[0].value)
at.sidebar.checkbox[1].check().run(); assert not at.exception, at.exception
print("HZ+matched ok | planets:", at.metric[0].value, "| warnings:", [w.value for w in at.warning])
at.sidebar.checkbox[0].uncheck().run(); at.sidebar.checkbox[1].uncheck().run()
at.sidebar.slider[1].set_value(100).run(); assert not at.exception, at.exception
print("distance<=100ly ok | planets:", at.metric[0].value)
at.sidebar.multiselect[0].set_value(["Direct imaging"]).run(); assert not at.exception, at.exception
print("imaging only ok | planets:", at.metric[0].value)
