To setup Gurobi using the WLS license:
1. `pip install gurobipy`
2. Create a Gurobi account using an `.edu` email, and generate a WLS license.
3. Note your **WLS Access ID**, **Secret Key**, and **License ID**
4. Put the following inside `~/gurobi.lic` file:
   - WLSACCESSID=your-access-id
   - WLSSECRET=your-secret-key
   - LICENSEID=your-license-id
5. You should be able to run Gurobi.