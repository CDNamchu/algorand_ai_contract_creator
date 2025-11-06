# Voting Contract Fixes Applied

## Date: 2025-11-06

## File Fixed
`outputs/contracts/contract_20251106_101313.py` (voting contract example)

## Critical Issues Fixed

### 1. Group Transaction Ordering Protection ✅
**Issue:** Contract assumed grouped transactions would always be in a specific order (app call first, asset transfer second) without enforcing it. Attackers could reorder transactions.

**Fix Applied:**
- Added `Txn.group_index() == Int(0)` assertion in `on_vote` and `on_withdraw`
- This explicitly requires the application call to be the first transaction in the group
- Prevents transaction reordering attacks

**Code Change:**
```python
# Before
Assert(
    And(
        Global.group_size() == Int(2),
        Gtxn[1].type_enum() == TxnType.AssetTransfer,
        ...
    )
)

# After
Assert(
    And(
        Global.group_size() == Int(2),
        Txn.group_index() == Int(0),  # ← NEW: Enforce ordering
        Gtxn[1].type_enum() == TxnType.AssetTransfer,
        ...
    )
)
```

### 2. Opt-In Validation ✅
**Issue:** Contract used `App.localGet()` and `App.localPut()` without verifying the account had opted into the application. This could cause runtime failures.

**Fix Applied:**
- Added `App.optedIn(Txn.sender(), Int(0))` check in `on_vote`
- Ensures account has opted in before attempting local state operations

**Code Change:**
```python
# Before
Assert(
    And(
        App.localGet(Txn.sender(), LOCAL_VOTED) == Int(0),
        ...
    )
)

# After
Assert(
    And(
        App.optedIn(Txn.sender(), Int(0)),  # ← NEW: Verify opt-in first
        App.localGet(Txn.sender(), LOCAL_VOTED) == Int(0),
        ...
    )
)
```

### 3. Closeout Protection ✅
**Issue:** Contract allowed users to close out at any time, even when they had locked tokens. This would delete their local state and make it impossible to withdraw their tokens.

**Fix Applied:**
- Changed CloseOut handler from unconditional approval to conditional
- Only allows closeout when `LOCAL_VOTE_AMOUNT == Int(0)` (no tokens locked)

**Code Change:**
```python
# Before
[
    Txn.on_completion() == OnComplete.CloseOut,
    Return(Int(1)),  # Always allowed
],

# After
[
    Txn.on_completion() == OnComplete.CloseOut,
    Seq([
        # Only allow closeout if no tokens are locked
        Assert(App.localGet(Txn.sender(), LOCAL_VOTE_AMOUNT) == Int(0)),
        Return(Int(1))
    ]),
],
```

## Generator Prompt Updated

Updated `src/algorand_ai_contractor/core/ai_engine.py` SYSTEM_PROMPT to include:

### New Guidance Added:
1. **GROUPED TRANSACTION REQUIREMENTS**
   - Always assert `Txn.group_index()` when using grouped transactions
   - Never assume transaction order without explicit checks
   - Example pattern provided

2. **OPT-IN AND STATE REQUIREMENTS**
   - Verify opt-in with `App.optedIn()` before local state operations
   - Document ASA opt-in requirements for application accounts
   - Always validate opt-in state before read/write

3. **CLOSEOUT PROTECTION**
   - Prevent closeout when users have non-zero locked balances
   - Example assertion pattern provided

## Verification

Contract compiles successfully to TEAL v6:
```bash
python outputs/contracts/contract_20251106_101313.py
# Output: Valid TEAL code with GroupIndex checks visible
```

## Remaining Known Issues (Not Fixed - Require Design Decisions)

These require manual review or deployment-time actions:

4. **Application ASA Opt-In** (Documentation Issue)
   - The application account must opt-in to the voting ASA before users can vote
   - **Action Required:** Add deployment instructions to opt-in the app account
   - **Alternative:** Add admin function to perform opt-in via inner transaction

5. **Vote Tallying** (Missing Feature)
   - Contract tracks individual votes but has no on-chain tally mechanism
   - **Recommendation:** Document that tallying happens off-chain via indexer
   - **Alternative:** Add vote choice tracking and tally function (gas-intensive)

6. **Round vs Timestamp** (Design Choice)
   - Uses rounds for deadline (VOTING_DURATION = 40320 rounds ≈ 7 days)
   - **Note:** Round timing can vary; timestamp-based deadlines more predictable
   - **Current:** Documented in code comments; acceptable for TestNet

## Testing Recommendations

1. **Group Order Test:** Create grouped transaction with reversed order, verify it fails
2. **Non-Opted Account Test:** Try voting without opt-in, verify assertion catches it
3. **Closeout While Locked Test:** Vote, then try closeout before withdrawal, verify it fails
4. **Normal Flow Test:** Opt-in → Vote (grouped) → Wait for deadline → Withdraw (grouped) → Closeout

## Commit Info
- Commit: ab476ab
- Branch: Syntax_fix
- Pushed to: origin/Syntax_fix

## Impact on Future Generations

With the updated AI prompt, future voting contracts (and similar grouped-transaction patterns) should automatically include:
- Group index assertions
- Opt-in validation
- Closeout protection for locked balances

Users should still manually verify these patterns in generated contracts and test thoroughly on TestNet before MainNet deployment.
