Building From Source
====================

Clone the Repository
---------------------
.. code-block:: bash

   git clone https://github.com/mlc-ai/web-llm.git
   cd web-llm

Install Dependencies
---------------------
.. code-block:: bash

   npm install

Build the Project
-----------------
.. code-block:: bash

   npm run build

Test Changes
------------

To test your changes, you can reuse an existing example or create a new example that specifically tests the new functionality you wish to provide.

To test the effects of your code change in an example, inside ``examples/<example>/package.json``, change ``"@mlc-ai/web-llm": "^0.2.xx"`` to ``"@mlc-ai/web-llm": "../.."`` to let it reference your local code. Note that sometimes you may need to switch between ``"file:../.."`` and ``"../.."`` to trigger npm to recognize new changes.

.. code-block:: bash

   cd examples/<example>
   # Modify package.json as described
   npm install
   npm start
