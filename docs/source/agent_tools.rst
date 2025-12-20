Using Agent
===========

``WeasToolkit`` exposes LangChain-compatible tools that let an agent inspect and edit a ``WeasWidget``.
One can use the `langchain-widget` package to create a chat agent with the toolkit. First, install the required packages:

.. code-block:: bash

   pip install langchain-widget[openai]


Then, you can create a toolkit instance by passing in a ``WeasWidget`` instance:

Here is an example of using the toolkit with a chat agent:

.. code-block:: python

   from weas_widget import WeasWidget, WeasToolkit
   from langchain_openai import ChatOpenAI
   from langchain_widget import LangChainWidget
   from langchain_openai import ChatOpenAI
   from dotenv import load_dotenv
   import ipywidgets as ipw

   load_dotenv()

   viewer = WeasWidget()

   chat_model = ChatOpenAI(model="gpt-4o-mini")
   chat = LangChainWidget(
      chat_model=chat_model,
      tools=WeasToolkit(viewer=viewer),
      title="WEAS Agent Chat",
      system_prompt=(
         "You are a scientific assistant. "
         "Use the available tools to inspect and manipulate the 3D structure."
      ),
      sidebar_open=False,
   )
   ipw.HBox([viewer, chat])

Extending the toolkit
---------------------

You can add your own tools in two ways:

1) Pass tools directly:

.. code-block:: python

   toolkit = WeasToolkit(viewer, extra_tools=[my_tool])
   tools = toolkit.tools

2) Register entry points under ``weas_widget.tools`` that return a tool or a tool factory.
