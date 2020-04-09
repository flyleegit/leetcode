

import java.util.*;

//一生不识N-Sum，刷尽天下也枉然
public class Sum {

    //https://leetcode.com/problems/two-sum/
    //basic 2sum
    public int[] twoSum(int[] nums, int target) {
        HashMap<Integer, Integer> set = new HashMap<Integer, Integer>();
        int[] res = new int[2];
        for (int i = 0; i < nums.length; i++) {
            set.put(nums[i], i);
        }

        for (int i = 0; i < nums.length; i++) {
            int tt = target - nums[i];
            if (set.containsKey(tt) && set.get(tt) != i) {
                res[0] = i;
                res[1] = set.get(tt);
                return res;
            }
        }
        return res;
    }

    //https://leetcode.com/problems/3sum/
    //basic 3sum
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        Arrays.sort(nums);
        for (int k = 0; k < nums.length - 2; k++) {
            int i = k + 1, j = nums.length - 1;

            if (nums[k] > 0) {
                break;
            }

            if (k > 0 && nums[k] == nums[k - 1]) {
                continue;
            }

            while (i < j && i < nums.length && j < nums.length) {
                int target = 0 - nums[k];
                if (nums[i] + nums[j] == target) {
                    List<Integer> tmpList = new ArrayList<Integer>();
                    tmpList.add(nums[k]);
                    tmpList.add(nums[i]);
                    tmpList.add(nums[j]);
                    res.add(new ArrayList<Integer>(tmpList));
                    while (i < nums.length - 1 && nums[i] == nums[i + 1]) {
                        i++;
                    }
                    while (j > 0 && nums[j] == nums[j - 1]) {
                        j--;
                    }
                    i++;
                    j--;
                } else if (nums[i] + nums[j] > target) {
                    j--;
                } else {
                    i++;
                }
            }
        }
        return res;
    }

    //https://leetcode.com/problems/3sum-closest/
    public int threeSumClosest(int[] nums, int target) {
        int res = nums[0] + nums[1] + nums[2];
        for (int i = 0; i < nums.length; i++) {
            for (int j = i + 1; j < nums.length; j++) {
                for (int k = j + 1; k < nums.length; k++) {
                    if (Math.abs(nums[i] + nums[j] + nums[k] - target) < Math.abs(res - target)) {
                        res = nums[i] + nums[j] + nums[k];
                    }
                }
            }
        }
        return res;
    }

    //https://leetcode.com/problems/4sum/
    //不知道这个4sum 有什么意思
    public List<List<Integer>> fourSum(int[] nums, int target) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        return res;
    }

    //https://leetcode.com/problems/letter-combinations-of-a-phone-number/
    public List<String> letterCombinations(String digits) {
        HashMap<Integer, char[]> map = new HashMap<Integer, char[]>();
        List<String> res = new ArrayList<String>();
        char[] a = {'a', 'b', 'c'};
        map.put(2, new char[]{'a', 'b', 'c'});
        map.put(3, new char[]{'d', 'e', 'f'});
        map.put(4, new char[]{'g', 'h', 'i'});
        map.put(5, new char[]{'j', 'k', 'l'});
        map.put(6, new char[]{'m', 'n', 'o'});
        map.put(7, new char[]{'p', 'q', 'r', 's'});
        map.put(8, new char[]{'t', 'u', 'v'});
        map.put(9, new char[]{'w', 'x', 'y', 'z'});
        String sb = new String();
        if (digits.isEmpty()) {
            return res;
        }
        letterCmbHelper(digits, sb, res, map);
        return res;
    }

    private void letterCmbHelper(String digits, String sb, List<String> res, HashMap<Integer, char[]> map) {
        if (digits.equals("")) {
            res.add(sb);
            return;
        }
        char[] charList = map.get(digits.charAt(0) - '0');
        for (char tt : charList) {
            letterCmbHelper(digits.substring(1), sb + tt, res, map);
        }
    }

    //https://leetcode.com/problems/remove-nth-node-from-end-of-list/
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode fast = new ListNode(-1);
        ListNode slow = new ListNode(-1);
        ListNode res = new ListNode(-1);
        fast.next = head;
        slow.next = head;
        res = slow;

        int step = 0;
        while (step < n) {
            fast = fast.next;
            step++;
        }

        while (fast.next != null) {
            fast = fast.next;
            slow = slow.next;
        }

        slow.next = slow.next.next;
        return res.next;
    }

    //https://leetcode.com/problems/generate-parentheses/
    public List<String> generateParenthesis(int n) {
        List<String> res = new ArrayList<String>();
        generateParentHelper(n, n, "", res);
        return res;
    }


    private void generateParentHelper(int left, int right, String out, List<String> res) {
        if (left == 0 && right == 0) {
            res.add(out);
            return;
        }

        if (left == right) {
            generateParentHelper(left - 1, right, out + "(", res);
            return;
        }

        if (left < right) {
            if (left > 0) {
                generateParentHelper(left - 1, right, out + "(", res);
            }
            generateParentHelper(left, right - 1, out + ")", res);
        }
    }

    //https://leetcode.com/problems/merge-k-sorted-lists/
    public ListNode mergeKLists(ListNode[] lists) {
        ListNode newNode = new ListNode(-1);
        ListNode res = newNode;
        int len = lists.length - 1;//小顶堆的长度
        int[] heap = new int[lists.length];
        HashMap<Integer, List<ListNode>> map = new HashMap<Integer, List<ListNode>>();
        for (int i = 0; i < lists.length; i++) {
            ListNode tmp = lists[i];
            if (tmp == null) {
                //有空链表
                len--;
                continue;
            }
            //k-v， listNode的值和对应的idx
            List<ListNode> tmpList = map.get(tmp.val);
            if (tmpList == null) {
                tmpList = new ArrayList<ListNode>();
            }
            tmpList.add(lists[i]);
            map.put(tmp.val, tmpList);

            heap[i] = tmp.val;

        }

        if (map.size() == 0) {
            return null;
        }

        while (len >= 0) {
            buildHeap(heap, len);
            //把最小的放到新的Node队列尾
            ListNode tmpNode = new ListNode(heap[0]);
            newNode.next = tmpNode;
            newNode = newNode.next;

            //找出刚才找出的值对应的ListNode
            List<ListNode> tt = map.get(heap[0]);
            ListNode nowNode = tt.get(0);
            //删除这个List中的这个ListNode
            tt.remove(0);
            //对应更新map
            if (tt.isEmpty()) {
                map.remove(heap[0]);
            } else {
                map.put(heap[0], tt);
            }

            //处理nowNode
            nowNode = nowNode.next;
            if (nowNode == null) {
                //其中有一个List已经找完
                heap[0] = heap[len];//把heap的最后一个提到第一个来，保证index在0~len是一个小顶堆
                len--;
                if (len < 0) {
                    break;
                }
                buildHeap(heap, len);
                continue;
            }
            int tmpVal = nowNode.val;
            List<ListNode> ll = map.get(tmpVal);
            if (ll == null) {
                ll = new ArrayList<ListNode>();
            }
            ll.add(nowNode);
            heap[0] = tmpVal;
            map.put(heap[0], ll);

            buildHeap(heap, len);
        }
        return res.next;
    }

    private void buildHeap(int[] nums, int end) {
        for (int i = (end - 1) / 2; i >= 0; i--) {
            heapHelper(nums, i, end);
        }
    }

    private void heapHelper(int[] nums, int idx, int end) {
//        if (idx > (end - 1) / 2) {
//            return;
//        }

        for (int i = idx; i >= 0; i--) {
            int left = 2 * idx + 1;
            int right = 2 * idx + 2;
            int min = idx;
            if (left <= end && nums[min] > nums[left]) {
                min = left;
            }

            if (right <= end && nums[min] > nums[right]) {
                min = right;
            }

            if (min != idx) {
                swap(nums, min, idx);
                heapHelper(nums, min, end);
            }
        }

    }
    //-------------------------------------------
    //-------------------------------------------
    //-------------------------------------------
    //-------------------------------------------

    public void swap(int[] arr, int i, int j) {
        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }

    //https://leetcode.com/problems/swap-nodes-in-pairs/
    public ListNode swapPairs(ListNode head) {
        ListNode res = new ListNode(-1);
        res.next = head;
        ListNode hh = res;
        ListNode nexth = new ListNode(-1);
        ListNode next1next = new ListNode(-1);

        while (hh != null && hh.next != null && hh.next.next != null) {
            head = hh.next;
            nexth = head.next;
            next1next = nexth.next;

            //变变变
            hh.next = nexth;
            nexth.next = head;
            head.next = next1next;

            hh = head;
        }
        return res.next;
    }


    //https://leetcode.com/problems/reverse-nodes-in-k-group/
    public ListNode reverseKGroup(ListNode head, int k) {
        ListNode dummy = new ListNode(-1);
        ListNode pre = dummy;
        ListNode cur = pre;
        dummy.next = head;
        int num = 0;//链表长度
        while (cur.next != null) {
            cur = cur.next;
            num++;
        }
        while (num >= k) {
            cur = pre.next;
            for (int i = 1; i < k; i++) {
                ListNode t = cur.next;
                cur.next = t.next;
                t.next = pre.next;
                pre.next = t;
            }
            pre = cur;
            num -= k;
        }
        return dummy.next;
    }


    public int removeDuplicates(int[] nums) {
        int i = 0, j = 0;
        while (j < nums.length) {
            if (nums[i] == nums[j]) {
                j++;
                continue;
            }
            swap(nums, ++i, j++);
        }
        return i + 1;
    }


    //https://leetcode.com/problems/remove-element/submissions/
    public int removeElement(int[] nums, int val) {
        int i = nums.length - 1, j = i;
        while (j >= 0) {
            if (nums[j] == val) {
                swap(nums, i--, j);
            }
            j--;
        }
        return i + 1;
    }

    //https://leetcode.com/problems/majority-element/
    public int majorityElement(int[] nums) {
        int major = nums[0];
        int cnt = 1;
        for (int i = 1; i < nums.length; i++) {
            if (cnt == 0) {
                major = nums[i];
                cnt = 1;
                continue;
            }

            if (major == nums[i]) {
                cnt++;
            } else {
                cnt--;
            }
        }

        return major;
    }

    //https://leetcode.com/problems/majority-element-ii/
    public List<Integer> majorityElement2(int[] nums) {
        List<Integer> res = new ArrayList<Integer>();
        int a = 0, b = 0, cnt1 = 0, cnt2 = 0, n = nums.length;
        for (int i = 0; i < n; i++) {
            int num = nums[i];
            if (num == a) {
                cnt1++;
            } else if (num == b) {
                cnt2++;
            } else if (cnt1 == 0) {
                a = num;
                cnt1 = 1;
            } else if (cnt2 == 0) {
                b = num;
                cnt2 = 1;
            } else {
                cnt1--;
                cnt2--;
            }
        }

        cnt1 = cnt2 = 0;
        for (int numa : nums) {
            if (numa == a) {
                cnt1++;
            } else if (numa == b) {
                cnt2++;
            }
        }
        if (cnt1 > n / 3) {
            res.add(cnt1);
        }

        if (cnt2 > n / 3) {
            res.add(cnt2);
        }
        return res;
    }

    //https://leetcode.com/problems/divide-two-integers/
    //一句话概括：m是除数，n是被除数，p是结果。把所有的除法运算转换成了2的运算
    // 循环算这一步：当m>2n时，p乘2,n乘2（位运算），
    //m = m-t   n ，继续上一步  然后结果相加
    public int divide(int dividend, int divisor) {
        long m = Math.abs(dividend), n = Math.abs(divisor), res = 0;
        if (m < n) {
            return 0;
        }
        long t = n, p = 1;
        while (m > (t << 1)) {
            t <<= 1;
            p <<= 1;
        }
        res += p + divide((int) (m - t), (int) n);
        if ((dividend < 0) ^ (divisor < 0)) {
            res = -res;
        }
        return res > Integer.MAX_VALUE ? Integer.MAX_VALUE : (int) res;
    }

    //https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/
//    solution:
//    https://www.programcreek.com/2016/08/leetcode-kth-smallest-element-in-a-sorted-matrix-java/
    public int kthSmallest(int[][] matrix, int k) {
        int m = matrix.length;

        int lower = matrix[0][0];
        int upper = matrix[m - 1][m - 1];

        while (lower < upper) {
            int mid = lower + ((upper - lower) >> 1);
            int count = count(matrix, mid);
            if (count < k) {
                lower = mid + 1;
            } else {
                upper = mid;
            }
        }

        return upper;
    }

    //找出target是matrix中第几小的数
    private int count(int[][] matrix, int target) {
        int m = matrix.length;
        int i = m - 1;
        int j = 0;
        int count = 0;

        while (i >= 0 && j < m) {

            if (matrix[i][j] <= target) {
                count += i + 1;
                j++;//往右(变大)
            } else {
                i--;//往上(变小)
            }
        }

        return count;
    }

    //https://leetcode.com/problems/substring-with-concatenation-of-all-words/
    public List<Integer> findSubstring(String s, String[] words) {
        HashMap<String, Integer> hashmap = new HashMap<String, Integer>();
        List<Integer> res = new ArrayList<Integer>();
        int len = 0;
        for (String str : words) {
            len = str.length();
            if (hashmap.containsKey(str)) {
                hashmap.put(str, hashmap.get(str) + 1);
            } else {
                hashmap.put(str, 1);
            }
        }
        if (len == 0) {
            return res;
        }

        for (int i = 0; i < s.length(); i++) {
            int j = i;
            HashMap<String, Integer> map = new HashMap(hashmap);
            while (!map.isEmpty()) {
                //j超出了s的范围
                if (j + len > s.length()) {
                    break;
                }
                String tmp = s.substring(j, j + len);

                //map中不包含这个string
                if (!map.containsKey(tmp)) {
                    break;
                }
                int cnt = map.get(tmp);

                //处理map中tmp的值
                if (cnt > 1) {
                    map.put(tmp, cnt - 1);
                } else {
                    map.remove(tmp);
                }

                j += len;
            }
            if (map.isEmpty()) {
                res.add(i);
            }
        }
        return res;
    }

    //https://leetcode.com/problems/permutations/
    public List<List<Integer>> permute(int[] nums) {
        int[] visited = new int[nums.length];
        List<Integer> out = new ArrayList<Integer>();
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        permuteHelper(nums, visited, out, res);
        return res;
    }

    public void permuteHelper(int[] nums, int[] visited, List<Integer> out, List<List<Integer>> res) {
        if (out.size() == nums.length) {
            res.add(new ArrayList<Integer>(out));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (visited[i] == 1) {
                continue;
            }
            visited[i] = 1;
            out.add(nums[i]);
            permuteHelper(nums, visited, out, res);
            visited[i] = 0;
            out.remove(out.size() - 1);
        }
    }

    //https://leetcode.com/problems/permutations-ii/
    public List<List<Integer>> permuteUnique(int[] nums) {
        Arrays.sort(nums);
        int[] visited = new int[nums.length];
        List<Integer> out = new ArrayList<Integer>();
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        permuteuniqueHelper(nums, visited, out, res);
        return res;

    }

    public void permuteuniqueHelper(int[] nums, int[] visited, List<Integer> out, List<List<Integer>> res) {
        if (out.size() == nums.length) {
            res.add(new ArrayList<Integer>(out));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (visited[i] == 1) {
                continue;
            }

            if (i > 0 && nums[i] == nums[i - 1] && visited[i - 1] == 0) {
                continue;
            }
            visited[i] = 1;
            out.add(nums[i]);
            permuteuniqueHelper(nums, visited, out, res);
            visited[i] = 0;
            out.remove(out.size() - 1);
        }
    }


    //    class Solution {
//        public:
//        void nextPermutation(vector<int>& nums) {
//        int n = nums.size(), i = n - 2, j = n - 1;
//            while (i >= 0 && nums[i] >= nums[i + 1]) --i;
//            if (i >= 0) {
//                while (nums[j] <= nums[i]) --j;
//                swap(nums[i], nums[j]);
//            }
//            reverse(nums.begin() + i + 1, nums.end());
//        }
//    };
    //https://leetcode.com/problems/next-permutation/
    public void nextPermutation(int[] nums) {
        int n = nums.length;
        int i = n - 2, j = n - 1;
        while (i >= 0 && nums[i] >= nums[i + 1]) {
            --i;
        }

        if (i >= 0) {
            while (nums[j] <= nums[i]) {
                j--;
            }
            swap(nums, i, j);
        }
    }

    //https://leetcode.com/problems/longest-valid-parentheses/
    //stack大法
    public int longestValidParentheses1(String s) {
        //真是难理解，还没有grand样那个好理解
        //官方stack答案
        Stack<Integer> stack = new Stack<Integer>();
        int res = 0;
        stack.push(-1);//无法用言语解释这一操作，就用例子"()"跑一边下面的代码就明白了
        for (int i = 0; i < s.length(); i++) {
            char tmp = s.charAt(i);
            if (tmp == '(') {
                stack.push(i);//左括号的index一定压进去
                continue;
            }
            //走到这里了，表明tmp一定是')'了

            stack.pop();//pop上一个fucking左括号的index(一定是左括号吗？不一定是)
            if (stack.isEmpty()) {
                //你看，把右括号怼进去了
                //原因就是这个右括号是个多余的，它的前面不是一个完整的匹配完整括号字符串
                stack.push(i);
            } else {
                //所以说，只有当stack pop完了之后，stack不为空，才表明此时的右括号是一个合理的括号字符串结尾
                res = Math.max(res, i - stack.peek());
            }
        }
        return res;
    }

    //遍历两遍String大法
    public int longestValidParentheses2(String s) {
        int left = 0;
        int right = 0;
        int res = 0;
        //从左往右
        for (int i = 0; i < s.length(); i++) {
            char tmp = s.charAt(i);

            if (tmp == '(') {
                left++;
            }

            if (tmp == ')') {
                right++;
            }

            if (left == right) {
                res = Math.max(left + right, res);
            } else if (right > left) {
                left = 0;
                right = 0;
            }
        }

        left = 0;
        right = 0;
        //从右往左
        for (int i = s.length() - 1; i >= 0; i--) {
            char tmp = s.charAt(i);

            if (tmp == '(') {
                left++;
            }

            if (tmp == ')') {
                right++;
            }

            if (left == right) {
                res = Math.max(left + right, res);
            } else if (left > right) {
                left = 0;
                right = 0;
            }
        }

        return res;
    }

    //DP大法
    public int longestValidParentheses3(String s) {
        int maxans = 0;
        int dp[] = new int[s.length()];
        for (int i = 1; i < s.length(); i++) {
            if (s.charAt(i) == ')') {
                if (s.charAt(i - 1) == '(') {
                    dp[i] = (i >= 2 ? dp[i - 2] : 0) + 2;
                } else if (i - dp[i - 1] > 0 && s.charAt(i - dp[i - 1] - 1) == '(') {
                    dp[i] = dp[i - 1] + ((i - dp[i - 1]) >= 2 ? dp[i - dp[i - 1] - 2] : 0) + 2;
                }
                maxans = Math.max(maxans, dp[i]);
            }
        }
        return maxans;
    }

    //https://leetcode.com/problems/search-in-rotated-sorted-array/
    public int search(int[] nums, int target) {
        int i = 0;
        int j = nums.length - 1;

        while (i <= j) {
            int mid = (i + j) / 2;
            if (nums[mid] == target) {
                return mid;
            }

            if (nums[i] < nums[mid]) {//左边有序
                if (nums[i] <= target && nums[mid] > target) {
                    j = mid - 1;
                } else {
                    i = mid + 1;
                }
            } else {//右边有序
                if (nums[mid] < target && nums[j] >= target) {
                    i = mid + 1;
                } else {
                    j = mid - 1;
                }
            }
        }
        return -1;
    }

    //https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/
    public int[] searchRange(int[] nums, int target) {
        int left = 0;
        int right = nums.length - 1;
        int[] res = {-1, -1};
        Boolean flag = false;
        int mid = 0;
        while (left <= right) {
            mid = (left + right) / 2;

            if (nums[mid] == target) {
                //发现了
                flag = true;
                break;
            }

            if (nums[mid] > target) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        left = right = -1;
        if (flag == true) {
            left = right = mid;
            while (left >= 0 && nums[left] == nums[mid]) {
                left--;

            }

            while (right <= nums.length - 1 && nums[right] == nums[mid]) {
                right++;

            }
            res[0] = left + 1;
            res[1] = right - 1;
        }

        return res;
    }

    //https://leetcode.com/problems/search-insert-position/
    public int searchInsert(int[] nums, int target) {
        int left = 0;
        int right = nums.length;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return left;
    }

    public boolean isValidSudoku(char[][] board) {
        // init data
        HashMap<Integer, Integer>[] rows = new HashMap[9];
        HashMap<Integer, Integer>[] columns = new HashMap[9];
        HashMap<Integer, Integer>[] boxes = new HashMap[9];
        for (int i = 0; i < 9; i++) {
            rows[i] = new HashMap<Integer, Integer>();
            columns[i] = new HashMap<Integer, Integer>();
            boxes[i] = new HashMap<Integer, Integer>();
        }

        // validate a board
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                char num = board[i][j];
                if (num != '.') {
                    int n = (int) num;
                    int box_index = (i / 3) * 3 + j / 3;

                    // keep the current cell value
                    rows[i].put(n, rows[i].getOrDefault(n, 0) + 1);
                    columns[j].put(n, columns[j].getOrDefault(n, 0) + 1);
                    boxes[box_index].put(n, boxes[box_index].getOrDefault(n, 0) + 1);

                    // check if this value has been already seen before
                    if (rows[i].get(n) > 1 || columns[j].get(n) > 1 || boxes[box_index].get(n) > 1)
                        return false;
                }
            }
        }

        return true;
    }

    //https://leetcode.com/problems/sudoku-solver/
    public void solveSudoku(char[][] board) {
        solveHelper(board, 0, 0);
    }

    public Boolean solveHelper(char[][] board, int row, int col) {
        if (row == 9) {
            return true;
        }
        for (int num = 1; num <= 9; num++) {

            if (!isValid(board, row, col, num)) {
                continue;
            }

            board[row][col] = (char) (num + 48);

            //solve next one
            int[] next = getNext(board, row, col);
            int nextRow = next[0];
            int nextCol = next[1];
            //已经走到最后一个
            if (nextRow == row && nextCol == col && board[row][col] != '.') {
                return true;
            }

            //solve next one
            if (solveHelper(board, nextRow, nextCol)) {
                return true;
            }

            board[row][col] = '.';
        }
        return false;
    }

    public int[] getNext(char[][] board, int row, int col) {
        int[] res = {row, col};
        int nextRow = row;
        int nextCol = col;
        if (row == 8 && col == 8) {
            return res;
        }


        if (col < 8) {
            nextCol = col + 1;
        } else {
            row = row + 1;
            col = 0;
        }
        //下一个已经有数字了
        if (board[row][col] != '.') {
            return getNext(board, nextRow, nextCol);
        }
        res[0] = nextRow;
        res[1] = nextCol;
        return res;
    }

    public Boolean isValid(char[][] board, int i, int j, int num) {
        for (int k = 0; k < 9; k++) {
            if (k == j) {
                continue;
            }
            char tmp = board[i][k];
            int nn = tmp - '0';
            if (nn == num) {
                return false;
            }
        }

        for (int k = 0; k < 9; k++) {
            if (k == i) {
                continue;
            }
            char tmp = board[k][j];
            int nn = tmp - '0';
            if (nn == num) {
                return false;
            }
        }
        int squareIdx = (i / 3) * 3 + j / 3;
        for (int x = (i / 3) * 3; x <= (i / 3) * 3 + 2; x++) {
            for (int y = (j / 3) * 3; y <= (j / 3) * 3 + 2; y++) {
                if (x == i && y == j) {
                    continue;
                }
                char tmp = board[x][y];
                int nn = tmp - '0';
                if (nn == num) {
                    return false;
                }
            }
        }
        return true;
    }

    //https://leetcode.com/problems/count-and-say/
    public String countAndSay(int n) {
        if (n == 1) {
            return "1";
        }
        String last = countAndSay(n - 1);
        int i = 0, j = i + 1;
        String res = "";
        while (j < last.length()) {
            if (last.charAt(j) == last.charAt(i)) {
                j++;
                continue;
            }
            res += String.valueOf(j - i) + String.valueOf(last.charAt(i));
            i = j;
            j++;
        }
        res += String.valueOf(j - i) + String.valueOf(last.charAt(i));
        return res;
    }

    //https://leetcode.com/problems/combination-sum/
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        List<Integer> out = new ArrayList<Integer>();
        int level = 0;
        combinationHelper(candidates, level, target, out, res);
        return res;
    }

    public void combinationHelper(int[] candidates, int level, int target, List<Integer> out, List<List<Integer>> res) {
        if (target < 0) {
            return;
        }
        if (target == 0) {
            res.add(new ArrayList<Integer>(out));
            return;
        }

        for (int i = level; i < candidates.length; i++) {
            int num = candidates[i];
            out.add(num);
            combinationHelper(candidates, i, target - num, out, res);
            out.remove(out.size() - 1);
        }
    }

    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        List<Integer> out = new ArrayList<Integer>();
        int level = 0;
        Arrays.sort(candidates);
        combinationHelper2(candidates, level, target, out, res);
        return res;
    }

    public void combinationHelper2(int[] candidates, int level, int target, List<Integer> out, List<List<Integer>> res) {
        if (target < 0) {
            return;
        }
        if (target == 0) {
            res.add(new ArrayList<Integer>(out));
            return;
        }
        int i = level;
        while (i < candidates.length) {
            int num = candidates[i];
            out.add(num);
            combinationHelper2(candidates, i + 1, target - num, out, res);
            out.remove(out.size() - 1);
            i++;
            while (i < candidates.length && i >= 1 && candidates[i] == candidates[i - 1]) {
                i++;
            }
        }
    }

    public int firstMissingPositive(int[] nums) {
        int len = nums.length;
        for (int i = 0; i < len; i++) {
            while (i < len && nums[i] <= len && nums[i] >= 1 && nums[nums[i] - 1] != nums[i]) {
                swap(nums, nums[i] - 1, i);
            }
        }
        int k = 0;
        for (; k < len; k++) {
            if (nums[k] != k + 1) {
                return k + 1;
            }
        }
        return k + 1;
    }

    public int firstMissingPositive2(int[] nums) {
        int res = 1;
        for (int i = 0; i < nums.length; i++) {
            while (nums[i] > 0 && nums[i] <= nums.length && (nums[nums[i] - 1] != nums[i])) {
                int t = nums[i] - 1;
                int tmp = nums[i];
                nums[i] = nums[t];
                nums[t] = tmp;
            }
        }
        int i = 0;
        for (i = 0; i < nums.length; i++) {
            if (nums[i] != i + 1) {
                return i + 1;
            }
        }
        return i + 1;
    }

    //https://leetcode.com/problems/trapping-rain-water/
    public int trap(int[] height) {
        int[] left = new int[height.length];
        int[] right = new int[height.length];

        int max = 0;

        for (int i = 0; i < height.length; i++) {
            if (max > height[i]) {
                left[i] = max;
            }
            max = Math.max(max, height[i]);
        }

        max = 0;
        for (int i = height.length - 1; i >= 0; i--) {
            if (max > height[i]) {
                right[i] = max;
            }
            max = Math.max(max, height[i]);
        }

        int res = 0;
        for (int i = 0; i < height.length; i++) {
            int ma = Math.min(left[i], right[i]);
            if (ma > height[i]) {
                res += ma - height[i];
            }
        }
        return res;
    }

    //https://leetcode.com/problems/container-with-most-water/
    public int maxArea(int[] height) {
        int left = 0, right = height.length - 1;
        int max = 0;
        while (left < right) {
            int area = 0;
            area = Math.min(height[left], height[right]) * (right - left);
            max = Math.max(area, max);
            if (height[right] > height[left]) {
                left++;
            } else {
                right--;
            }
        }
        return max;
    }

    //https://leetcode.com/problems/multiply-strings/
    public String multiply(String num1, String num2) {
        String res = "";
        for (int i = num1.length() - 1; i >= 0; i--) {
            int leftMod = 0;
            String nowOut = new String();
            for (int j = num2.length() - 1; j >= 0; j--) {
                char tmp1 = num1.charAt(i);
                char tmp2 = num2.charAt(j);
                int tmpMulti = (tmp1 - '0') * (tmp2 - '0');

                nowOut = (tmpMulti + leftMod) % 10 + nowOut;
                leftMod = (tmpMulti + leftMod) / 10;
            }

            if (leftMod > 0) {
                nowOut = leftMod + nowOut;
            }
            String zeroAdd = "";
            for (int tt = 0; tt < num1.length() - i - 1; tt++) {
                zeroAdd += "0";
            }
            res = stringAdd(res, nowOut + zeroAdd);
            int a = 0;
        }
        int idx = 0;
        for (; idx < res.length() - 1; idx++) {
            if (res.charAt(idx) != '0') {
                break;
            }
        }

        return res.substring(idx);
    }

    public String stringAdd(String str1, String str2) {

        String longer = str1.length() > str2.length() ? str1 : str2;
        String shorter = str1.length() > str2.length() ? str2 : str1;

        int shorterLen = shorter.length();
        for (int i = 0; i < longer.length() - shorterLen; i++) {
            shorter = "0" + shorter;
        }

        String res = "";
        int leftMod = 0;
        for (int i = longer.length() - 1; i >= 0; i--) {
            char tmp1 = longer.charAt(i);
            char tmp2 = shorter.charAt(i);

            int tmpAdd = (tmp1 - '0') + (tmp2 - '0');

            res = (tmpAdd + leftMod) % 10 + res;

            leftMod = (tmpAdd + leftMod) / 10;
        }

        if (leftMod > 0) {
            res = leftMod + res;
        }
        return res;
    }

    //https://leetcode.com/problems/wildcard-matching/
    public boolean isMatch(String s, String p) {
        int m = s.length(), n = p.length();
        Boolean[][] dp = new Boolean[m + 1][n + 1];

        for (int i = 0; i <= m; i++) {
            for (int j = 0; j <= n; j++) {
                dp[i][j] = false;
            }
        }
        dp[0][0] = true;

        for (int i = 1; i <= n; ++i) {
            if (p.charAt(i - 1) == '*') {
                dp[0][i] = dp[0][i - 1];
            }
        }

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (p.charAt(j - 1) == '*') {
                    dp[i][j] = dp[i - 1][j] || dp[i][j - 1];
                } else {
                    dp[i][j] = (s.charAt(i - 1) == p.charAt(j - 1) || p.charAt(j - 1) == '?') && dp[i - 1][j - 1];
                }
            }
        }
        return dp[m][n];
    }

    public boolean isMatch2(String s, String p) {
        int i = 0, j = 0, iStar = -1, jStar = -1, m = s.length(), n = p.length();
        while (i < m) {
            if (j < n && (s.charAt(i) == p.charAt(j) || p.charAt(j) == '?')) {
                i++;
                j++;
            } else if (j < n && p.charAt(j) == '*') {
                iStar = i;
                jStar = j++;
            } else if (iStar >= 0) {
                i = ++iStar;
                j = jStar + 1;
            } else {
                return false;
            }
        }
        while (j < n && p.charAt(j) == '*') {
            j++;
        }
        return j == n;
    }


    //https://leetcode.com/problems/regular-expression-matching/
    public boolean isMatchR(String text, String pattern) {
        if (pattern.isEmpty()) {
            return text.isEmpty();
        }
        boolean first_match = (!text.isEmpty() &&
                (pattern.charAt(0) == text.charAt(0) || pattern.charAt(0) == '.'));

        if (pattern.length() >= 2 && pattern.charAt(1) == '*') {
            return (isMatchR(text, pattern.substring(2)) || // e.g :text = "aab", pattern="c*a*b"
                    (first_match && isMatchR(text.substring(1), pattern))); // e.g :text = "abc", pattern=".*"
        } else {
            return first_match && isMatchR(text.substring(1), pattern.substring(1));//e.g:text = "aab", pattern="a*b"
        }
    }

    enum Result {
        TRUE, FALSE,
    }

    Result[][] memo;

    public boolean isMatchR2(String text, String pattern) {
        memo = new Result[text.length() + 1][pattern.length() + 1];
        return dp(0, 0, text, pattern);
    }

    public boolean dp(int i, int j, String text, String pattern) {
        if (memo[i][j] != null) {
            return memo[i][j] == Result.TRUE;
        }
        boolean ans;
        if (j == pattern.length()) {
            ans = i == text.length();
        } else {
            boolean first_match = (i < text.length() &&
                    (pattern.charAt(j) == text.charAt(i) ||
                            pattern.charAt(j) == '.'));

            if (j + 1 < pattern.length() && pattern.charAt(j + 1) == '*') {
                ans = (dp(i, j + 2, text, pattern) ||
                        first_match && dp(i + 1, j, text, pattern));
            } else {
                ans = first_match && dp(i + 1, j + 1, text, pattern);
            }
        }
        memo[i][j] = ans ? Result.TRUE : Result.FALSE;
        return ans;
    }

    //https://leetcode.com/problems/jump-game/
    public boolean canJump(int[] nums) {
        int len = nums.length;
        int now = 0;
        int maxIdx = 0;
        while (now < len) {
            //maxIdx都到不了now这里
            if (maxIdx < now) {
                break;
            }
            maxIdx = maxIdx > now + nums[now] ? maxIdx : now + nums[now];
            if (maxIdx >= len - 1) {
                break;
            }
            now++;
        }

        if (maxIdx >= len - 1) {
            return true;
        }
        return false;
    }

    //https://leetcode.com/problems/jump-game-ii/
    public int jump(int[] nums) {
        int len = nums.length;
        int[] dp = new int[len];
        for (int i = 0; i < dp.length; i++) {
            dp[i] = len + 1;
        }
        dp[0] = 0;
        for (int i = 0; i < nums.length; i++) {
            int val = nums[i];
            for (int j = i; j <= i + val && j < len; j++) {
                dp[j] = ((dp[i] + 1) < dp[j]) ? dp[i] + 1 : dp[j];
            }
        }
        return dp[len - 1];
    }

    //https://leetcode.com/problems/rotate-image/
    public void rotate(int[][] matrix) {
        for (int i = 0; i < matrix.length; i++) {
            for (int j = i; j < matrix[0].length; j++) {
                if (i == j) {
                    continue;
                }
                int tmp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = tmp;
            }
        }

        for (int k = 0; k < matrix.length; k++) {
            int i = 0, j = matrix[0].length - 1;
            while (i <= j) {
                int tmp = matrix[k][i];
                matrix[k][i] = matrix[k][j];
                matrix[k][j] = tmp;
                i++;
                j--;
            }
        }
    }

    //https://leetcode.com/problems/group-anagrams/
    public List<List<String>> groupAnagrams(String[] strs) {
        HashMap<String, List<String>> hashmap = new HashMap<String, List<String>>();
        List<List<String>> res = new ArrayList<List<String>>();
        for (String str : strs) {
            char[] charArr = str.toCharArray();
            Arrays.sort(charArr);

            String newStr = new String(charArr);

            List<String> tmp = hashmap.getOrDefault(newStr, new ArrayList<String>());
            tmp.add(str);
            hashmap.put(newStr, tmp);
        }

        for (String key : hashmap.keySet()) {
            res.add(hashmap.get(key));
        }
        return res;
    }

    //https://leetcode.com/problems/powx-n/
    public double myPow(double x, int n) {
        if (x == 0) {
            return 0;
        }

        if (n == 0) {
            return 1;
        }

        return n > 0 ? x * myPow(x, Math.abs(n) - 1) : 1 / (x * myPow(x, Math.abs(n) - 1));
    }

    //https://leetcode.com/problems/sqrtx/
    public int mySqrt(int x) {
        if (x < 0) {
            return -1;
        }
        for (int i = 0; i <= x; i++) {
            if (i * i == x) {
                return i;
            }
            if (i * i > x) {
                return i - 1;
            }
        }
        return -1;
    }

//    int mySqrt(int x) {
//        if (x <= 1) return x;
//        int left = 0, right = x;
//        while (left < right) {
//            int mid = left + (right - left) / 2;
//            if (x / mid >= mid) left = mid + 1;
//            else right = mid;
//        }
//        return right - 1;
//    }

    public int mySqrt2(int x) {
        if (x <= 1) {
            return x;
        }
        int left = 0, right = x;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (x / mid >= mid) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        return right - 1;
    }

    //https://leetcode.com/problems/n-queens/
    public List<List<String>> solveNQueens(int n) {
        String[][] out = new String[n][n];
        List<List<String>> res = new ArrayList<List<String>>();
        NQueenHelper(n, out, 0, 0, res);
        return res;
    }

    public void NQueenHelper(int n, String[][] out, int row, int col, List<List<String>> res) {
        if (row == n) {
            List<String> tmp = new ArrayList<String>();
            int QCnt = 0;
            for (int i = 0; i < n; i++) {
                String str = String.join("", out[i]);
                tmp.add(str);
                for (int j = 0; j < n; j++) {
                    if (out[i][j] == "Q") {
                        QCnt++;
                    }
                }
            }
            if (QCnt == n) {
                res.add(tmp);
            }
            return;
        }

        //当走到一行的最后一个，直接这么写到9的时候会超时，因为会出现一行没有Q，就进入了无效循环
//        if (col == n) {
//            NQueenHelper(n, out, row + 1, 0, res);
//            return;
//        }

        if (col == n) {
            //这一行里面一定有Q
            int qq = 0;
            for (int l = 0; l < n; l++) {
                if (out[row][l] == "Q") {
                    qq++;
                }
            }
            //这一行至少有一个Q的时候才继续下一行的循环
            if (qq >= 1) {
                NQueenHelper(n, out, row + 1, 0, res);
            }
            return;
        }


        //自己写的逻辑
        out[row][col] = "Q";
        if (QueenValid(out, row, col)) {
            NQueenHelper(n, out, row, col + 1, res);
        }
        out[row][col] = ".";
        NQueenHelper(n, out, row, col + 1, res);
    }

    public Boolean QueenValid(String[][] out, int row, int col) {
        int i = row - 1;
        int j = col;
        while (i >= 0) {
            if (out[i][col] == "Q") {
                return false;
            }
            i--;
        }

        j = col - 1;
        while (j >= 0) {
            if (out[row][j] == "Q") {
                return false;
            }
            j--;
        }

        i = row - 1;
        j = col - 1;
        while (i >= 0 && j >= 0) {
            if (out[i][j] == "Q") {
                return false;
            }
            i--;
            j--;
        }

        i = row - 1;
        j = col + 1;
        while (i >= 0 && j < out.length) {
            if (out[i][j] == "Q") {
                return false;
            }
            i--;
            j++;
        }
        return true;
    }

    //https://leetcode.com/problems/maximum-subarray/
    //Input: [-2,1,-3,4,-1,2,1,-5,4],
    //Output: 6
    //Explanation: [4,-1,2,1] has the largest sum = 6.
    public int maxSubArray(int[] nums) {
        int len = nums.length;
        int max = nums[0];
        int[] dp = new int[len];
        dp[0] = nums[0];
        for (int i = 1; i < len; i++) {
            dp[i] = Math.max(dp[i - 1] + nums[i], nums[i]);
            max = Math.max(max, dp[i]);
        }
        return max;
    }

    //https://leetcode.com/problems/spiral-matrix/
    public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> res = new ArrayList<Integer>();
        int row = matrix.length;
        if (row == 0) {
            return res;
        }
        int col = matrix[0].length;

        int left = 0, right = col - 1, up = 0, down = row - 1;

        while (left <= right || up <= down) {

            //从左到右
            int i = up, j = left;
            while (j <= right) {
                res.add(matrix[i][j]);
                j++;
            }
            up++;
            if (up > down) {
                break;
            }

            //从上到下
            i = up;
            j = right;
            while (i <= down) {
                res.add(matrix[i][j]);
                i++;
            }
            right--;

            if (left > right) {
                break;
            }

            //从右到左
            i = down;
            j = right;
            while (j >= left) {
                res.add(matrix[i][j]);
                j--;
            }
            down--;
            if (up > down) {
                break;
            }

            //从下到上
            i = down;
            j = left;
            while (i >= up) {
                res.add(matrix[i][j]);
                i--;
            }
            left++;
            if (left > right) {
                break;
            }
        }
        return res;
    }

    //https://leetcode.com/problems/length-of-last-word/
    public int lengthOfLastWord(String s) {
        s = s.trim();
        int len = 0;
        for (int i = s.length() - 1; i >= 0; i--) {

            char tmp = s.charAt(i);

//            if (i == s.length() - 1 && tmp == ' ') {
//                continue;
//            }

            if (tmp == ' ') {
                break;
            }
            len++;
        }
        return len;
    }

    //https://leetcode.com/problems/rotate-list/
    public ListNode rotateRight(ListNode head, int k) {
        ListNode res = new ListNode(-1);
        res.next = head;
        ListNode fast = head;
        ListNode slow = head;

        //计算head长度
        int len = 0;
        while (fast != null) {
            fast = fast.next;
            len++;
        }
        if (len == 0) {
            return head;
        }
        k = k % len;

        if (k <= 0) {
            return head;
        }

        fast = head;
        while (k > 0) {
            fast = fast.next;
            k--;
        }


        while (fast.next != null) {
            fast = fast.next;
            slow = slow.next;
        }

        res = slow.next;
        slow.next = null;
        fast.next = head;
        return res;
    }

    //https://leetcode.com/problems/unique-paths/
    public int uniquePaths(int m, int n) {
        int[][] dp = new int[m][n];
        for (int i = 0; i < m; i++) {
            dp[i][0] = 1;
        }

        for (int j = 0; j < n; j++) {
            dp[0][j] = 1;
        }

        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
        return dp[m - 1][n - 1];
    }

    //https://leetcode.com/problems/unique-paths-ii/
    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        int m = obstacleGrid.length;
        int n = obstacleGrid[0].length;
        int[][] dp = new int[m][n];
        if (obstacleGrid[0][0] == 1) {
            return 0;
        }
        dp[0][0] = 1;
        for (int i = 1; i < m; i++) {
            if (obstacleGrid[i][0] == 1) {
                dp[i][0] = 0;
            } else {
                dp[i][0] = dp[i - 1][0];
            }
        }

        for (int j = 1; j < n; j++) {
            if (obstacleGrid[0][j] == 1) {
                dp[0][j] = 0;
            } else {
                dp[0][j] = dp[0][j - 1];
            }
        }

        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (obstacleGrid[i][j] == 1) {
                    dp[i][j] = 0;
                    continue;
                }
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
        return dp[m - 1][n - 1];
    }

    //https://leetcode.com/problems/minimum-path-sum/
    public int minPathSum(int[][] grid) {
        int row = grid.length;
        int col = grid[0].length;

        int[][] dp = new int[row][col];

        dp[0][0] = grid[0][0];

        for (int i = 1; i < row; i++) {
            dp[i][0] = dp[i - 1][0] + grid[i][0];
        }

        for (int j = 1; j < col; j++) {
            dp[0][j] = dp[0][j - 1] + grid[0][j];
        }

        for (int i = 1; i < row; i++) {
            for (int j = 1; j < col; j++) {
                dp[i][j] = Math.min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
            }
        }
        return dp[row - 1][col - 1];
    }

    //https://leetcode.com/problems/simplify-path/
    public String simplifyPath(String path) {

        String[] pathArr = path.split("/");

        List<String> res = new ArrayList<String>();

        for (String str : pathArr) {

            if (str.equals(".") || str.equals(" ") || str.equals("")) {
                continue;
            }

            if (str.equals("..")) {
                if (res.size() > 0) {
                    res.remove(res.size() - 1);
                }
                continue;
            }

            res.add(str);
        }

        String outStr = "/";
        for (int i = 0; i < res.size(); i++) {
            outStr += res.get(i) + "/";
        }

        //去掉最后一个"/"
        if (outStr.length() > 1) {
            return outStr.substring(0, outStr.length() - 1);
        }

        return outStr;
    }

    //https://leetcode.com/problems/edit-distance/
    public int minDistance(String word1, String word2) {
        int m = word1.length();
        int n = word2.length();

        int[][] dp = new int[m + 1][n + 1];

        //翻译翻译
        //从word1转为空的情况，只能全做删除
        for (int i = 0; i <= m; i++) {
            dp[i][0] = i;
        }

        //翻译翻译
        //从空转为word2的情况，只能一个一个加
        for (int j = 0; j <= n; j++) {
            dp[0][j] = j;
        }

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                //翻译翻译
                //如果word1的当前字符等于word2的当前字符，那他们的转换次数与上一个字符的次数相等
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    //dp[i - 1][j - 1] + 1 代表修改
                    //dp[i - 1][j] + 1 代表删除
                    //dp[i][j-1] + 1 代表插入
                    //细细的品味下
                    dp[i][j] = Math.min(dp[i - 1][j - 1], Math.min(dp[i - 1][j], dp[i][j - 1])) + 1;
                }
            }
        }

        return dp[m][n];
    }

    public void setZeroes(int[][] matrix) {
        int row = matrix.length;
        int col = matrix[0].length;
        int[][] visited = new int[row][col];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                if (visited[i][j] == 0 && matrix[i][j] == 0) {
                    setZero(i, j, matrix, visited);
                }
            }
        }
        return;
    }

    public void setZero(int row, int col, int[][] matrix, int[][] visited) {

        for (int i = 0; i < matrix.length; i++) {

            //这句话是为了防止这类case
            //[0,1,2,0]
            //[3,4,5,2]
            //[1,3,1,5]
            if (matrix[i][col] != 0) {
                visited[i][col] = 1;
            }
            matrix[i][col] = 0;
        }

        for (int i = 0; i < matrix[0].length; i++) {
            if (matrix[row][i] != 0) {
                visited[row][i] = 1;
            }
            matrix[row][i] = 0;
        }
    }

    //https://leetcode.com/problems/search-a-2d-matrix/
    public boolean searchMatrix(int[][] matrix, int target) {
        if (matrix.length <= 0) {
            return false;
        }
        if (matrix[0].length <= 0) {
            return false;
        }
        int left = 0;
        int right = matrix.length - 1;
        while (left <= right) {
            int mid = (left + right) / 2;
            if (matrix[mid][0] == target) {
                return true;
            }

            if (matrix[mid][0] > target) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        if (right < 0 || right > matrix.length - 1) {
            return false;
        }
        int row = right;
        left = 0;
        right = matrix[row].length - 1;

        while (left <= right) {
            int mid = (left + right) / 2;
            if (matrix[row][mid] == target) {
                return true;
            }

            if (matrix[row][mid] > target) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return false;
    }

    //https://leetcode.com/problems/sort-colors/
    public void sortColors(int[] nums) {
        for (int i = 0; i < nums.length; i++) {
            for (int j = 1; j < nums.length; j++) {
                if (nums[j] < nums[j - 1]) {
                    swap(nums, j - 1, j);
                }
            }
        }
    }


    public void sortColors2(int[] nums) {
        int left = 0;
        int curr = 0;
        int right = nums.length - 1;
        while (curr <= right) {
            if (nums[curr] == 0) {
                swap(nums, left, curr);
                left++;
                curr++;
                continue;
            }

            if (nums[curr] == 1) {
                curr++;
                continue;
            }

            if (nums[curr] == 2) {
                swap(nums, curr, right);
                right--;
                continue;
            }
        }
    }

    //https://leetcode.com/problems/move-zeroes/
    public void moveZeroes(int[] nums) {
        int slow = 0;
        int fast = 0;

        while (slow < nums.length && nums[slow] != 0) {
            slow++;
            fast = slow;
        }

        while (fast < nums.length) {
            if (nums[fast] != 0) {
                swap(nums, fast, slow);
                slow++;
            }
            fast++;
        }
        return;
    }

    public String JustFuckIt(String str) {
        char[] inputArr = str.toCharArray();
        int charStart = 0;
        int charEnd = 0;
        int numStart = 0;
        int numEnd = 0;
        int i = 0;
        for (i = 0; i < inputArr.length; i++) {
            char tmp = inputArr[i];
            if (!Character.isDigit(tmp)) {
                break;
            }
        }
        charStart = i;
        for (; i < inputArr.length; i++) {
            //交换number和 char
            while (i < inputArr.length && Character.isDigit(inputArr[i])) {
                fuckSwap(inputArr, charStart, i);
                charStart++;
                i++;
            }
        }
        return new String(inputArr);
    }

    public void fuckSwap(char[] input, int i, int j) {
        for (int tmp = j; tmp > i; tmp--) {
            swap(input, tmp, tmp - 1);
        }
    }

    public void swap(char[] input, int i, int j) {
        char tmp = input[i];
        input[i] = input[j];
        input[j] = tmp;
    }

    //https://leetcode.com/problems/minimum-window-substring/
    public String minWindow(String s, String t) {

        if (s.length() == 0 || t.length() == 0) {
            return "";
        }

        // Dictionary which keeps a count of all the unique characters in t.
        Map<Character, Integer> dictT = new HashMap<Character, Integer>();
        for (int i = 0; i < t.length(); i++) {
            int count = dictT.getOrDefault(t.charAt(i), 0);
            dictT.put(t.charAt(i), count + 1);
        }

        // Number of unique characters in t, which need to be present in the desired window.
        int required = dictT.size();

        // Left and Right pointer
        int l = 0, r = 0;

        // formed is used to keep track of how many unique characters in t
        // are present in the current window in its desired frequency.
        // e.g. if t is "AABC" then the window must have two A's, one B and one C.
        // Thus formed would be = 3 when all these conditions are met.
        int formed = 0;

        // Dictionary which keeps a count of all the unique characters in the current window.
        Map<Character, Integer> windowCounts = new HashMap<Character, Integer>();

        // ans list of the form (window length, left, right)
        int[] ans = {-1, 0, 0};

        while (r < s.length()) {
            // Add one character from the right to the window
            char c = s.charAt(r);
            int count = windowCounts.getOrDefault(c, 0);
            windowCounts.put(c, count + 1);

            // If the frequency of the current character added equals to the
            // desired count in t then increment the formed count by 1.
            if (dictT.containsKey(c) && windowCounts.get(c).intValue() == dictT.get(c).intValue()) {
                formed++;
            }

            // Try and contract the window till the point where it ceases to be 'desirable'.
            while (l <= r && formed == required) {
                c = s.charAt(l);
                // Save the smallest window until now.
                if (ans[0] == -1 || r - l + 1 < ans[0]) {
                    ans[0] = r - l + 1;
                    ans[1] = l;
                    ans[2] = r;
                }

                // The character at the position pointed by the
                // `Left` pointer is no longer a part of the window.
                windowCounts.put(c, windowCounts.get(c) - 1);
                if (dictT.containsKey(c) && windowCounts.get(c).intValue() < dictT.get(c).intValue()) {
                    formed--;
                }

                // Move the left pointer ahead, this would help to look for a new window.
                l++;
            }

            // Keep expanding the window once we are done contracting.
            r++;
        }

        return ans[0] == -1 ? "" : s.substring(ans[1], ans[2] + 1);
    }

    //https://leetcode.com/problems/combinations/
    public List<List<Integer>> combine(int n, int k) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        List<Integer> out = new ArrayList<Integer>();
        HashSet<Integer> visited = new HashSet<Integer>();
        combineHelper(n, k, 1, visited, out, res);
        return res;
    }

    public void combineHelper(int n, int k, int level, HashSet<Integer> visited, List<Integer> out, List<List<Integer>> res) {
        if (out.size() == k) {
            res.add(new ArrayList<Integer>(out));
            return;
        }

        for (int i = level; i <= n; i++) {
            if (visited.contains(i)) {
                continue;
            }
            out.add(i);
            visited.add(i);
            combineHelper(n, k, i + 1, visited, out, res);
            out.remove(out.size() - 1);
            visited.remove(i);
        }
    }

    //https://leetcode.com/problems/subsets/
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        List<Integer> out = new ArrayList<Integer>();
        subsetsHelper(nums, 0, out, res);
        return res;
    }

    public void subsetsHelper(int[] nums, int level, List<Integer> out, List<List<Integer>> res) {

        res.add(new ArrayList<Integer>(out));

        if (out.size() == nums.length) {
            return;
        }

        for (int i = level; i < nums.length; i++) {
            out.add(nums[i]);
            subsetsHelper(nums, i + 1, out, res);
            out.remove(out.size() - 1);
        }
    }

    //https://leetcode.com/problems/word-search/
    public boolean exist(char[][] board, String word) {
        int[][] visited = new int[board.length][board[0].length];
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                if (existHelper(board, i, j, word, visited)) {
                    return true;
                }
            }
        }
        return false;
    }

    public Boolean existHelper(char[][] board, int row, int col, String word, int[][] visited) {
        if (word.length() <= 0) {
            return true;
        }

        if (row < 0 || row > board.length - 1
                || col < 0 || col > board[row].length - 1) {
            return false;
        }

        if (board[row][col] != word.charAt(0)) {
            return false;
        }

        if (visited[row][col] == 1) {
            return false;
        }
        visited[row][col] = 1;

        //右边
        Boolean rightFlag = existHelper(board, row, col + 1, word.substring(1), visited);
        if (rightFlag) {
            return rightFlag;
        }


        //下边
        Boolean downFlag = existHelper(board, row + 1, col, word.substring(1), visited);
        if (downFlag) {
            return downFlag;
        }

        //左边
        Boolean leftFlag = existHelper(board, row, col - 1, word.substring(1), visited);
        if (leftFlag) {
            return leftFlag;
        }

        //上边
        Boolean upFlag = existHelper(board, row - 1, col, word.substring(1), visited);
        if (upFlag) {
            return upFlag;
        }
        visited[row][col] = 0;

        return false;
    }

    //https://leetcode.com/problems/remove-duplicates-from-sorted-array-ii/
    public int removeDuplicates2(int[] nums) {
        int i = 1, j = 1;
        int cnt = 1;

        while (j < nums.length) {
            if (nums[j] == nums[j - 1]) {
                cnt++;
            } else {
                cnt = 1;
            }

            if (cnt <= 2) {
                nums[i] = nums[j];
                i++;
            }
            j++;
        }
        return i;
    }


    public int removeDuplicates3(int[] nums) {

        //
        // Initialize the counter and the second pointer.
        //
        int j = 1, count = 1;

        //
        // Start from the second element of the array and process
        // elements one by one.
        //
        for (int i = 1; i < nums.length; i++) {

            //
            // If the current element is a duplicate, increment the count.
            //
            if (nums[i] == nums[i - 1]) {

                count++;

            } else {

                //
                // Reset the count since we encountered a different element
                // than the previous one.
                //
                count = 1;
            }

            //
            // For a count <= 2, we copy the element over thus
            // overwriting the element at index "j" in the array
            //
            if (count <= 2) {
                nums[j++] = nums[i];
            }
        }
        return j;
    }

    //https://leetcode.com/problems/search-in-rotated-sorted-array-ii/
    public boolean search22(int[] nums, int target) {
        int left = 0;
        int right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                return true;
            }
            if (nums[left] == nums[mid]) {
                left++;
                continue;
            }

            if (nums[right] == nums[mid]) {
                right--;
                continue;
            }

            if (nums[left] < nums[mid]) {//左边递增
                if (nums[left] <= target && nums[mid] > target) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            } else {
                if (nums[right] >= target && nums[mid] < target) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
        }
        return false;
    }

    //https://leetcode.com/problems/remove-duplicates-from-sorted-list-ii/
    public ListNode deleteDuplicates(ListNode head) {
        if (head == null) {
            return head;
        }
        ListNode res = new ListNode(-1);
        res.next = head;
        ListNode yeah = res;
        while (yeah.next != null && yeah.next.next != null) {
            ListNode slow = yeah.next;
            ListNode fast = yeah.next.next;

            if (slow.val == fast.val) {
                while (fast != null && slow.val == fast.val) {
                    fast = fast.next;
                }
                yeah.next = fast;
            } else {
                yeah = yeah.next;
            }
        }
        return res.next;
    }

    //https://leetcode.com/problems/largest-rectangle-in-histogram/
    public int largestRectangleArea(int[] heights) {
        List<Integer> tmpMax = new ArrayList<Integer>();
        for (int i = 0; i < heights.length; i++) {
            if ((i == heights.length - 1) || heights[i] > heights[i + 1]) {
                tmpMax.add(i);
            }
        }
        int res = 0;
        for (int i = 0; i < tmpMax.size(); i++) {
            int key = tmpMax.get(i);
            int minH = heights[key];

            for (int j = key; j >= 0; j--) {
                minH = Math.min(minH, heights[j]);
                int area = minH * (key - j + 1);
                res = Math.max(res, area);
            }
        }
        return res;
    }

    public int largestRectangleArea2(int[] heights) {
        int maxarea = 0;
        for (int i = 0; i < heights.length; i++) {
            int minheight = Integer.MAX_VALUE;
            for (int j = i; j < heights.length; j++) {
                //从i到j中最矮的那个柱子才是这个区间的短板
                minheight = Math.min(minheight, heights[j]);
                //这个区间的最小是就是短板乘以这个区间的长度啊啊啊啊啊啊啊  => minheight * (j-i +1)
                maxarea = Math.max(maxarea, minheight * (j - i + 1));
            }
        }
        return maxarea;
    }

    //https://leetcode.com/problems/maximal-rectangle/
    public int maximalRectangle(char[][] matrix) {
        int row = matrix.length;
        int col = matrix[0].length;
        int[][][] dp = new int[row][col][2];
        if (matrix[0][0] == '1') {
            dp[0][0] = new int[]{1, 1};
        }

        for (int i = 1; i < col; i++) {
            char tmp = matrix[0][i];
            if (tmp == '1') {
                dp[0][i] = new int[]{dp[0][i - 1][0] + 1, 1};
            }
        }

        for (int i = 1; i < row; i++) {
            char tmp = matrix[i][0];
            if (tmp == '1') {
                dp[i][0] = new int[]{1, dp[i - 1][0][1] + 1};
            }
        }
        int res = 0;

        for (int i = 1; i < row; i++) {
            for (int j = 1; j < col; j++) {
                char tmp = matrix[i][j];
                if (tmp == '1') {
                    dp[i][j][0] = dp[i][j - 1][0] + 1;
                    dp[i][j][1] = dp[i - 1][j][1] + 1;
                }
                res = Math.max(dp[i][j][0] * dp[i][j][1], res);
            }
        }
        return res;
    }

    public int maximalRectangle2(char[][] matrix) {

        if (matrix.length == 0) return 0;
        int maxarea = 0;
        int[][] dp = new int[matrix.length][matrix[0].length];

        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                if (matrix[i][j] == '1') {

                    // compute the maximum width and update dp with it
                    dp[i][j] = j == 0 ? 1 : dp[i][j - 1] + 1;

                    int width = dp[i][j];

                    // compute the maximum area rectangle with a lower right corner at [i, j]
                    for (int k = i; k >= 0; k--) {
                        width = Math.min(width, dp[k][j]);
                        maxarea = Math.max(maxarea, width * (i - k + 1));
                    }
                }
            }
        }
        return maxarea;
    }

    //https://leetcode.com/problems/partition-list/
    public ListNode partition(ListNode head, int x) {
        ListNode left = new ListNode(-1);
        ListNode leftTmp = left;

        ListNode right = new ListNode(-1);
        ListNode rightTmp = right;

        ListNode tmp = head;
        while (tmp != null) {
            int val = tmp.val;
            if (val < x) {
                leftTmp.next = new ListNode(val);
                leftTmp = leftTmp.next;
            } else {
                rightTmp.next = new ListNode(val);
                rightTmp = rightTmp.next;
            }
            tmp = tmp.next;
        }

        leftTmp.next = right.next;
        return left.next;
    }

    //https://leetcode.com/problems/scramble-string/
    //这是一个错误的答案,260 / 283 test cases passed.
    //比如s1="abcd",s2="cabd",会return false
    public boolean isScramble(String s1, String s2) {
        if (s1.equals(s2)) {
            return true;
        }
        if (s1.length() == 2) {
            if (s1.charAt(0) == s2.charAt(1) && s1.charAt(1) == s2.charAt(0)) {
                return true;
            } else {
                return false;
            }
        }

        //找到切分的点
        //这个思路有问题，切分的点不一定是这个index,
        // 比如s1="abcd",s2="cabd",用下面的思路，会从a这个切分点找， 会return false
        //实际上，以c为切分点，得return true
        int index = 0;
        for (; index < s1.length(); index++) {
            if (s2.charAt(index) == s1.charAt(0) && index != s1.length() - 1
                    && isScramble(s1.substring(0, index + 1), s2.substring(0, index + 1))
                    && isScramble(s1.substring(index + 1), s2.substring(index + 1))) {
                return true;
            }

            if (s2.charAt(index) == s1.charAt(0) && index != 0
                    && isScramble(s1.substring(0, s1.length() - index), s2.substring(index))
                    && isScramble(s1.substring(s1.length() - index), s2.substring(0, index))) {
                return true;
            }

        }
        return false;
    }

    public boolean isScramble2(String s1, String s2) {
        if (s1.equals(s2)) {
            return true;
        }

        if (s1.length() == 2) {
            if (s1.charAt(0) == s2.charAt(1) && s1.charAt(1) == s2.charAt(0)) {
                return true;
            } else {
                return false;
            }
        }

        for (int index = 0; index < s1.length(); index++) {
            if (index != s1.length() - 1
                    && isScramble(s1.substring(0, index + 1), s2.substring(0, index + 1))
                    && isScramble(s1.substring(index + 1), s2.substring(index + 1))) {
                return true;
            }

            if (index != 0
                    && isScramble(s1.substring(0, s1.length() - index), s2.substring(index))
                    && isScramble(s1.substring(s1.length() - index), s2.substring(0, index))) {
                return true;
            }
        }
        return false;
    }

    //https://leetcode.com/problems/merge-sorted-array/
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int length = m + n;
        while (m >= 1 && n >= 1) {
            if (nums1[m - 1] >= nums2[n - 1]) {
                nums1[m + n - 1] = nums1[m - 1];
                m--;
            } else {
                nums1[m + n - 1] = nums2[n - 1];
                n--;
            }
        }
        while (n >= 1) {
            nums1[n - 1] = nums2[n - 1];
            n--;
        }
    }
}
