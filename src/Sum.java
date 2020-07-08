

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

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

    //https://leetcode.com/problems/subsets-ii/
//    Input: [1,2,2]
//    Output:
//            [
//            [2],
//            [1],
//            [1,2,2],
//            [2,2],
//            [1,2],
//            []
//            ]
    /*
    input:[1,2,3]
    [1],
    [2],
     */
    public List<List<Integer>> subsetsWithDup(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        List<Integer> out = new ArrayList<Integer>();
        int k = 0;
        subsetWithDupHelper(nums, k, out, res);
        return res;
    }

    public void subsetWithDupHelper(int[] nums, int level, List<Integer> out, List<List<Integer>> res) {


        res.add(new ArrayList<Integer>(out));
        if (out.size() == nums.length) {
            return;
        }

        for (int i = level; i < nums.length; i++) {
            out.add(nums[i]);
            subsetWithDupHelper(nums, i + 1, out, res);
            out.remove(out.size() - 1);

            while (i + 1 < nums.length && nums[i] == nums[i + 1]) {
                i++;
            }
        }
    }

    //https://leetcode.com/problems/decode-ways/
    // 超时啊啊啊啊啊，太长了就超时
    public int numDecodings(String s) {
        List<String> out = new ArrayList<String>();
        List<List<String>> res = new ArrayList<List<String>>();
        numDecodeHelper(s, out, res);
        return res.size();
    }

    public void numDecodeHelper(String s, List<String> out, List<List<String>> res) {
        if (s.length() == 0) {
            res.add(new ArrayList<String>(out));
            return;
        }

        //1个
        String tmp = s.substring(0, 1);
        int foo = Integer.parseInt(tmp);
        if (foo == 0) {
            return;
        }

        out.add(tmp);
        numDecodeHelper(s.substring(1), out, res);
        out.remove(out.size() - 1);

        if (s.length() >= 2) {
            tmp = s.substring(0, 2);
            foo = Integer.parseInt(tmp);
            if (foo > 26) {
                return;
            }
            out.add(tmp);
            numDecodeHelper(s.substring(2), out, res);
            out.remove(out.size() - 1);
        }
    }

    public int numDecodings2(String s) {
        if (s.isEmpty() || s.charAt(0) == '0') return 0;
        int[] dp = new int[s.length() + 1];
        dp[0] = 1;
        for (int i = 1; i < dp.length; ++i) {
            dp[i] = (s.charAt(i - 1) == '0') ? 0 : dp[i - 1];
            if (i > 1 && (s.charAt(i - 2) == '1' || (s.charAt(i - 2) == '2' && s.charAt(i - 1) <= '6'))) {
                dp[i] += dp[i - 2];
            }
        }
        return dp[s.length()];
    }

    //https://leetcode.com/problems/reverse-linked-list-ii/
    public ListNode reverseBetween(ListNode head, int m, int n) {
        ListNode res = new ListNode(-1);
        res.next = head;
        ListNode slow = res;

        int cnt = 0;
        while (cnt < m - 1) {
            slow = slow.next;
            cnt++;
        }
        ListNode curr = slow.next;

        //开始类似于插入的形式变化
        while (cnt < n - 1 && curr.next != null) {
            ListNode tmp = slow.next;//注意，curr会一直往后走，所以curr不恒等于slow.next
            //example :  1->2->3->4->5
            //变成1->3->2->4->5
            slow.next = curr.next;
            curr.next = curr.next.next;
            slow.next.next = tmp;
            cnt++;
        }
        return res.next;
    }

    //https://leetcode.com/problems/restore-ip-addresses/submissions/
    public List<String> restoreIpAddresses(String s) {
        List<String> out = new ArrayList<String>();
        List<String> res = new ArrayList<String>();

        restoreIpHelper(s, out, res);
        return res;
    }

    public void restoreIpHelper(String s, List<String> out, List<String> res) {

        if (out.size() == 4 && !s.equals("")) {
            return;
        }

        if (s.equals("") && out.size() != 4) {
            return;
        }

        if (s.equals("") && out.size() == 4) {
            String tmp = "";
            for (String qq : out) {
                tmp += qq + ".";
            }
            tmp = tmp.substring(0, tmp.length() - 1);
            res.add(tmp);
            return;
        }


        //1个
        out.add(s.substring(0, 1));
        restoreIpHelper(s.substring(1), out, res);
        out.remove(out.size() - 1);

        //开头为0的
        //不可能出现01.122.123.123
        //但是可以出现0.1.2.3
        if (s.startsWith("0")) {
            return;
        }

        //2个
        if (s.length() >= 2) {

            out.add(s.substring(0, 2));
            restoreIpHelper(s.substring(2), out, res);
            out.remove(out.size() - 1);
        }

        //3个
        if (s.length() >= 3) {
            String tmp = s.substring(0, 3);
            if (Integer.parseInt(tmp) > 255) {
                return;
            }
            out.add(s.substring(0, 3));
            restoreIpHelper(s.substring(3), out, res);
            out.remove(out.size() - 1);
        }
    }

    //https://leetcode.com/problems/binary-tree-inorder-traversal/
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<Integer>();
        inorderHelper(root, res);
        return res;
    }

    public void inorderHelper(TreeNode root, List<Integer> res) {
        if (root == null) {
            return;
        }
        inorderHelper(root.left, res);
        res.add(root.val);
        inorderHelper(root.right, res);
    }

    //https://leetcode.com/problems/unique-binary-search-trees-ii/
    public List<TreeNode> generateTrees(int n) {
        if (n == 0) {
            return new LinkedList<TreeNode>();
        }
        return generate_trees(1, n);
    }

    public LinkedList<TreeNode> generate_trees(int start, int end) {
        LinkedList<TreeNode> all_trees = new LinkedList<TreeNode>();
        if (start > end) {
            all_trees.add(null);
            return all_trees;
        }

        // pick up a root
        for (int i = start; i <= end; i++) {
            // all possible left subtrees if i is choosen to be a root
            LinkedList<TreeNode> left_trees = generate_trees(start, i - 1);

            // all possible right subtrees if i is choosen to be a root
            LinkedList<TreeNode> right_trees = generate_trees(i + 1, end);

            // connect left and right trees to the root i
            for (TreeNode l : left_trees) {
                for (TreeNode r : right_trees) {
                    TreeNode current_tree = new TreeNode(i);
                    current_tree.left = l;
                    current_tree.right = r;
                    all_trees.add(current_tree);
                }
            }
        }
        return all_trees;
    }

    public int numTrees(int n) {
        List<TreeNode> res = new ArrayList<TreeNode>();
        return numTreesHelper(1, n).size();
    }

    public List<TreeNode> numTreesHelper(int start, int end) {
        List<TreeNode> res = new ArrayList<TreeNode>();
        if (start > end) {
            res.add(null);
            return res;
        }

        for (int i = start; i <= end; i++) {
            ListNode root = new ListNode(i);
            List<TreeNode> leftList = numTreesHelper(start, i - 1);
            List<TreeNode> rightList = numTreesHelper(i + 1, end);

            for (TreeNode left : leftList) {
                for (TreeNode right : rightList) {
                    TreeNode tmp = new TreeNode(root.val);
                    tmp.left = left;
                    tmp.right = right;
                    res.add(tmp);
                }
            }
        }
        return res;
    }

    public int numTrees2(int n) {
        int[] G = new int[n + 1];
        G[0] = 1;
        G[1] = 1;

        for (int i = 2; i <= n; ++i) {
            for (int j = 1; j <= i; ++j) {
                G[i] += G[j - 1] * G[i - j];
            }
        }
        return G[n];
    }

    //https://leetcode.com/problems/same-tree/
    public boolean isSameTree(TreeNode p, TreeNode q) {
        if (p == null && q == null) {
            return true;
        }

        if (p == null || q == null) {
            return false;
        }

        return p.val == q.val && isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
    }

    //https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        //先序:  根->左->右
        //中序：左->根->右
        if (preorder.length == 0) {
            return null;
        }
        int rootVal = preorder[0];
        TreeNode root = new TreeNode(rootVal);
        int index; //中序遍历root的index
        for (index = 0; index < inorder.length; index++) {
            if (inorder[index] == rootVal) {
                break;
            }
        }

        root.left = buildTree(Arrays.copyOfRange(preorder, 1, index + 1), Arrays.copyOfRange(inorder, 0, index));
        root.right = buildTree(Arrays.copyOfRange(preorder, index + 1, preorder.length), Arrays.copyOfRange(inorder, index + 1, inorder.length));
        return root;
    }

    //https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        Queue<TreeNode> queue = new LinkedList<TreeNode>();
        if (root == null) {
            return res;
        }

        queue.offer(root);
        int cnt = 0;
        while (!queue.isEmpty()) {
            List<Integer> out = new ArrayList<Integer>();
            for (int i = queue.size(); i > 0; i--) {
                TreeNode tmp = queue.poll();
                out.add(tmp.val);
                if (tmp.left != null) {
                    queue.offer(tmp.left);
                }
                if (tmp.right != null) {
                    queue.offer(tmp.right);
                }
            }
            List<Integer> tmp = new ArrayList<Integer>();

            if (cnt % 2 == 1) {
                for (int i = out.size() - 1; i >= 0; i--) {
                    tmp.add(out.get(i));
                }
            } else {
                tmp = new ArrayList<Integer>(out);
            }
            res.add(tmp);
            cnt++;
        }
        return res;
    }

    //https://leetcode.com/problems/interleaving-string/
    //这方法超时，超时，超时，超时
    public boolean isInterleave(String s1, String s2, String s3) {
        if (s1.isEmpty() && s2.isEmpty() && s3.isEmpty()) {
            return true;
        }
        //s3 为空，但是其他两个不为空，返回false
        if (s3.isEmpty()) {
            return false;
        }
        if (!s1.isEmpty() && s1.charAt(0) == s3.charAt(0)) {
            if (isInterleave(s1.substring(1), s2, s3.substring(1))) {
                return true;
            }
        }

        if (!s2.isEmpty() && s2.charAt(0) == s3.charAt(0)) {
            if (isInterleave(s1, s2.substring(1), s3.substring(1))) {
                return true;
            }
        }
        return false;
    }

    public boolean isInterleave2(String s1, String s2, String s3) {
        int memo[][] = new int[s1.length()][s2.length()];
        for (int i = 0; i < s1.length(); i++) {
            for (int j = 0; j < s2.length(); j++) {
                memo[i][j] = -1;
            }
        }
        return is_Interleave(s1, 0, s2, 0, s3, 0, memo);
    }

    public boolean is_Interleave(String s1, int i, String s2, int j, String s3, int k, int[][] memo) {
        if (i == s1.length()) {
            return s2.substring(j).equals(s3.substring(k));
        }
        if (j == s2.length()) {
            return s1.substring(i).equals(s3.substring(k));
        }
        if (memo[i][j] >= 0) {
            return memo[i][j] == 1 ? true : false;
        }
        boolean ans = false;
        if (s3.charAt(k) == s1.charAt(i) && is_Interleave(s1, i + 1, s2, j, s3, k + 1, memo)
                || s3.charAt(k) == s2.charAt(j) && is_Interleave(s1, i, s2, j + 1, s3, k + 1, memo)) {
            ans = true;
        }
        memo[i][j] = ans ? 1 : 0;
        return ans;
    }

    //https://leetcode.com/problems/recover-binary-search-tree/
    //官方默认推荐的O(n)额外空间，当然后面的follow up官方希望我们用O(1)的额外空间
    //支持各种调换顺序，只要这个搜索树调换过顺序，都可以用这个方法
    public void recoverTree(TreeNode root) {
        List<TreeNode> list = new ArrayList<TreeNode>();
        List<Integer> vals = new ArrayList<Integer>();
        inorder(root, list, vals);
        Collections.sort(vals);
        for (int i = 0; i < list.size(); i++) {
            list.get(i).val = vals.get(i);
        }
    }

    public void inorder(TreeNode root, List<TreeNode> list, List<Integer> vals) {
        if (root == null) {
            return;
        }

        inorder(root.left, list, vals);
        list.add(root);
        vals.add(root.val);
        inorder(root.right, list, vals);
    }

    //https://leetcode.com/problems/symmetric-tree/
    //层次遍历，然后每一层看是不是镜子
    public boolean isSymmetric(TreeNode root) {
        Queue<TreeNode> queue = new LinkedList<TreeNode>();
        queue.add(root);
        while (!queue.isEmpty()) {
            List<Integer> list = new ArrayList<Integer>();
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                TreeNode tmp = queue.poll();
                if (tmp != null) {
                    queue.offer(tmp.left);
                    queue.offer(tmp.right);
                    list.add(tmp.val);
                } else {
                    list.add(null);
                }

            }
            int i = 0, j = list.size() - 1;
            while (i <= j) {
                if (list.get(i++) != list.get(j--)) {
                    return false;
                }
            }
        }
        return true;
    }

    //递归
    public boolean isSymmetric2(TreeNode root) {
        return isMirror(root, root);
    }

    public boolean isMirror(TreeNode t1, TreeNode t2) {
        if (t1 == null && t2 == null) return true;
        if (t1 == null || t2 == null) return false;
        return (t1.val == t2.val)
                && isMirror(t1.right, t2.left)
                && isMirror(t1.left, t2.right);
    }

    //类似于sameTree的队列
    public boolean isSymmetric3(TreeNode root) {
        Queue<TreeNode> q = new LinkedList<>();
        q.add(root);
        q.add(root);
        while (!q.isEmpty()) {
            TreeNode t1 = q.poll();
            TreeNode t2 = q.poll();
            if (t1 == null && t2 == null) continue;
            if (t1 == null || t2 == null) return false;
            if (t1.val != t2.val) return false;
            q.add(t1.left);
            q.add(t2.right);
            q.add(t1.right);
            q.add(t2.left);
        }
        return true;
    }

    //https://leetcode.com/problems/maximum-depth-of-binary-tree/
    public int maxDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
    }

    //https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/
    public TreeNode sortedArrayToBST(int[] nums) {
        return sortedArrayHelper(nums, 0, nums.length - 1);
    }

    public TreeNode sortedArrayHelper(int[] nums, int start, int end) {

        if (start > end) {
            return null;
        }
        int mid = start + (end - start) / 2;

        TreeNode root = new TreeNode(nums[mid]);

        root.left = sortedArrayHelper(nums, start, mid - 1);

        root.right = sortedArrayHelper(nums, mid + 1, end);
        return root;
    }

    //https://leetcode.com/problems/convert-sorted-list-to-binary-search-tree/
    public TreeNode sortedListToBST(ListNode head) {
        if (head == null) {
            return null;
        }
        ListNode pre = new ListNode(-1);
        pre.next = head;
        ListNode midLink = head;
        ListNode fast = head;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            pre = midLink;
            midLink = midLink.next;
        }

        TreeNode root = new TreeNode(midLink.val);
        pre.next = null;
        root.right = sortedListToBST(midLink.next);
        if (midLink == head) {
            head = null;
        }
        root.left = sortedListToBST(head);
        return root;
    }

    //https://leetcode.com/problems/minimum-depth-of-binary-tree/
    public int minDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }

        if (root.left == null) {
            return 1 + minDepth(root.right);
        }

        if (root.right == null) {
            return 1 + minDepth(root.left);
        }
        return 1 + Math.min(minDepth(root.left), minDepth(root.right));
    }

    //https://leetcode.com/problems/path-sum/
    public boolean hasPathSum(TreeNode root, int sum) {
        if (root == null) {
            return false;
        }
        int rootVal = root.val;

        if (rootVal == sum && root.left == null && root.right == null) {
            return true;
        }
        return hasPathSum(root.left, sum - rootVal) || hasPathSum(root.right, sum - rootVal);
    }

    //https://leetcode.com/problems/path-sum-ii/
    public List<List<Integer>> pathSum(TreeNode root, int sum) {
        List<Integer> out = new ArrayList<Integer>();
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        pathSumHelper(root, sum, out, res);
        return res;

    }

    public void pathSumHelper(TreeNode root, int sum, List<Integer> out, List<List<Integer>> res) {
        if (root == null) {
            return;
        }
        int rootVal = root.val;
        if (rootVal == sum && root.left == null && root.right == null) {
            out.add(root.val);
            res.add(new ArrayList<Integer>(out));
            out.remove(out.size() - 1);
            return;
        }
        out.add(rootVal);
        pathSumHelper(root.left, sum - rootVal, out, res);
        pathSumHelper(root.right, sum - rootVal, out, res);
        out.remove(out.size() - 1);
    }

    //https://www.cnblogs.com/grandyang/p/9615871.html
    //https://leetcode-cn.com/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof/solution/mian-shi-ti-36-er-cha-sou-suo-shu-yu-shuang-xian-5/
    TreeNode pre, head;

    public TreeNode treeToDoublyList(TreeNode root) {
        if (root == null) {
            return null;
        }
        dfs(root);
        //这两句话把这个链表变成了一个环，可能题目这么要求？
        head.left = pre;
        pre.right = head;
        return head;
    }

    //TreeNode没法引用传递，需要一个全局变量
    public void dfs(TreeNode cur) {
        if (cur == null) {
            return;
        }
        //中序遍历，先找到最左的叶子节点
        dfs(cur.left);

        //pre表示迭代后双向链表的尾节点
        if (pre != null) {
            pre.right = cur;
        } else {
            //head表示变成双向链表的头节点
            //当pre为空，表明这是整棵树的最左边那个叶子节点，
            head = cur;
        }
        cur.left = pre;
        pre = cur;

        dfs(cur.right);
    }

    //https://leetcode.com/problems/flatten-binary-tree-to-linked-list/
    //what i wanna say is "F***k"
    //二叉树先序遍历变变变变变变变
    public void flatten(TreeNode root) {
        if (root == null) {
            return;
        }

        flatten(root.left);

        flatten(root.right);

        TreeNode tmp = root.right;
        root.right = root.left;
        root.left = null;
        while (root.right != null) {
            root = root.right;
        }
        root.right = tmp;
    }

    //https://leetcode.com/problems/distinct-subsequences/
    public int numDistinct(String s, String t) {
        //递推公式
        //dp[i][j] = dp[i][j - 1] + (T[i - 1] == S[j - 1] ? dp[i - 1][j - 1] : 0)

        int[][] dp = new int[t.length() + 1][s.length() + 1];

        //空字符串是任何字符串的子串
        for (int j = 0; j < s.length() + 1; j++) {
            dp[0][j] = 1;
        }

        for (int i = 1; i < t.length() + 1; i++) {
            for (int j = 1; j < s.length() + 1; j++) {

                if (t.charAt(i - 1) == s.charAt(j - 1)) {

                    dp[i][j] = dp[i - 1][j - 1] + dp[i][j - 1];
                } else {
                    //因为s新增的一位字符不等于t新增的一位，
                    //所以并没有对子字符串带来新的可能性
                    //所以个数就以左边那个值为准(s减少了一位，t没变)
                    dp[i][j] = dp[i][j - 1];
                }
            }
        }
        return dp[t.length()][s.length()];
    }

    //https://leetcode.com/problems/populating-next-right-pointers-in-each-node/
    public Node connect(Node root) {
        if (root == null) {
            return root;
        }
        Queue<Node> queue = new LinkedList<Node>();
        queue.offer(root);
        Node pre = new Node();
        while (queue.size() > 0) {

            int size = queue.size();
            int i = size - 1;

            while (i >= 0) {
                Node tmp = queue.poll();

                if (i == (size - 1)) {
                    pre = tmp;
                } else {
                    pre.next = tmp;
                    pre = pre.next;

                }
                if (tmp.left != null) {
                    queue.offer(tmp.left);
                    queue.offer(tmp.right);
                }
                i--;
            }
        }
        return root;
    }

    //https://leetcode.com/problems/populating-next-right-pointers-in-each-node-ii/
    //不一样的地方在于，这个可能是个非完全二叉树
    public Node connect2(Node root) {
        if (root == null) {
            return root;
        }
        Queue<Node> queue = new LinkedList<Node>();
        queue.offer(root);
        Node pre = new Node();
        while (queue.size() > 0) {

            int size = queue.size();
            int i = size - 1;

            while (i >= 0) {
                Node tmp = queue.poll();

                if (i == (size - 1)) {
                    pre = tmp;
                } else {
                    pre.next = tmp;
                    pre = pre.next;

                }
                if (tmp.left != null) {
                    queue.offer(tmp.left);
                }

                if (tmp.right != null) {
                    queue.offer(tmp.right);
                }
                i--;
            }
        }
        return root;
    }

    //https://leetcode.com/problems/triangle/
    public int minimumTotal(List<List<Integer>> triangle) {
        int row = triangle.size();
        int res = 0;
        if (row <= 0) {
            return res;
        }
        int[][] dp = new int[row][triangle.get(row - 1).size()];
        dp[0][0] = triangle.get(0).get(0);
        for (int tmp = 1; tmp < dp.length; tmp++) {
            dp[tmp][0] = dp[tmp - 1][0] + triangle.get(tmp).get(0);
        }
        res = dp[row - 1][0];
        for (int i = 1; i < dp.length; i++) {

            for (int j = 1; j < triangle.get(i).size(); j++) {
                if (j > i - 1) {//此时i,j在最右边
                    dp[i][j] = dp[i - 1][j - 1] + triangle.get(i).get(j);
                } else {
                    dp[i][j] = Math.min(dp[i - 1][j - 1], dp[i - 1][j]) + triangle.get(i).get(j);
                }

                if (i == row - 1) {
                    res = Math.min(res, dp[i][j]);
                }
            }
        }
        return res;
    }

    //https://leetcode.com/problems/best-time-to-buy-and-sell-stock/
    public int maxProfit(int[] prices) {
        int minprice = Integer.MAX_VALUE;
        int maxprofit = 0;
        for (int i = 0; i <= prices.length; i++) {
            if (prices[i] < minprice) {
                minprice = prices[i];
            } else if ((prices[i] - minprice) > maxprofit) {
                maxprofit = prices[i] - minprice;
            }
        }
        return maxprofit;
    }

    //另一种解法
//    public int maxProfit(int[] prices) {
//        int res = 0, buy = Integer.MAX_VALUE;
//        for (int i = 0; i < prices.length; i++) {
//            int price = prices[i];
//            buy = Math.min(buy, price);
//            res = Math.max(res, price - buy);
//        }
//        return res;
//    }

    //https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/
    public int maxProfit2(int[] prices) {
        int maxprofit = 0;
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] > prices[i - 1])
                maxprofit += prices[i] - prices[i - 1];
        }
        return maxprofit;
    }

    //https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/
    //有点难，后面再看
//    public int maxProfit3(int[] prices) {
//        if (prices.length == 0) {
//            return 0;
//        }
//
//        int n = prices.length;
//        int[][] g = new int[n][3];//到达第i天时最多可进行j此交易的最大利润，全局
//
//        int[][] l = new int[n][3];//到达第i天时最多可进行j次交易，并且最后一次交易在最后一天卖出的最大利润（局部最优解）
//
//        //递推公式
//        /**
//         //局部最优解是相比前一天交易
//         l[i][j] = max(g[i-1][j-1]) + max(diff,0),l[i-1][j] + diff)
//         g[i][j] = max(l[i][j],g[i-1][j])
//         */
//
//        for (int i = 1; i < prices.length; i++) {
//            int diff = prices[i] - prices[i - 1];
//            for (int j = 1; j <= 2; j++) {
//                l[i][j] = Math.max(g[i - 1][j - 1] + Math.max(diff, 0), l[i - 1][j] + diff);
//
//                g[i][j] = Math.max(l[i][j], g[i - 1][j]);
//            }
//        }
//
//        return g[n-1][2]
//    }

    //https://leetcode.com/problems/binary-tree-maximum-path-sum/

    int maxDepthRes = Integer.MIN_VALUE;//初始化全局变量，结果为一个最小值

    public int maxPathSum(TreeNode root) {
        maxPathSumHelper(root);
        return maxDepthRes;
    }

    //注意，第一点：以root为根节点的二叉树，计算最大path，可能经过root节点，也可能不经过root节点

    //解释解释，解释这个function
    //这个function是计算 以root为根节点，且经过root节点的path的最大值
    //注意这个function的return是不能以root为根节点形成半环的
    //也就是说  只能是  max(function(root.left),function(root.right)) + root.val

    //所以在计算的过程中，有个全局变量maxDepthRes来算最终的最大值，
    // 因为：最大path，可能经过root节点，也可能不经过root节点
    public int maxPathSumHelper(TreeNode root) {
        if (root == null) {
            return 0;
        }

        //递归调用左子树的最大值
        int left = Math.max(maxPathSumHelper(root.left), 0);

        ////递归调用右子树的最大值
        int right = Math.max(maxPathSumHelper(root.right), 0);

        //left + right + root.val  这个很有意思
        //这个值就是以root为衔接点的半环最大值
        maxDepthRes = Math.max(maxDepthRes, left + right + root.val);

        return Math.max(left, right) + root.val;
    }

    //https://leetcode.com/problems/valid-palindrome/
    public boolean isPalindrome(String s) {
        int l = 0;
        int r = s.length() - 1;
        while (l <= r) {
            char ll = Character.toLowerCase(s.charAt(l));
            char rr = Character.toLowerCase(s.charAt(r));

            //不是a~z
            if (ll < 97 || ll > 122) {
                l++;
                continue;
            }

            //不是a~z
            if (rr < 97 || rr > 122) {
                r--;
                continue;
            }


            if (ll != rr) {
                return false;
            }
            l++;
            r--;
        }
        return true;
    }

    //https://leetcode.com/problems/word-ladder/
    //DFS 这方法超时~  超时，超时，超时
    int minLen = Integer.MAX_VALUE;

    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
        int len = wordList.size();
        int[] visited = new int[len];
        ladderLengthHelper(beginWord, endWord, visited, wordList, 0);
        return minLen == Integer.MAX_VALUE ? 0 : minLen + 1;
    }

    public void ladderLengthHelper(String word1, String endWord, int[] visited, List<String> wordList, int cnt) {
        if (word1.equals(endWord)) {
            minLen = Math.min(minLen, cnt);
            return;
        }
        int len = wordList.size();

        for (int i = 0; i < len; i++) {
            if (visited[i] == 1) {
                continue;
            }
            String word = wordList.get(i);

            if (!compareMatch(word1, word)) {
                continue;
            }
            visited[i] = 1;
            ladderLengthHelper(word, endWord, visited, wordList, cnt + 1);
            visited[i] = 0;
        }
    }

    //判断两个字符串是不是就差一个char
    public boolean compareMatch(String word1, String word2) {
        int len = word1.length();
        int cnt = 0;
        for (int i = 0; i < len; i++) {

            if (word1.charAt(i) != word2.charAt(i)) {
                cnt++;
            }

            if (cnt >= 2) {
                return false;
            }
        }
        return true;
    }

    //BFS, 核心逻辑是用a~z一个一个去替换单词中的字母，可能换到某一个就中了
    //不用DFS一路走到黑
    public int ladderLength2(String beginWord, String endWord, List<String> wordList) {
        HashSet<String> set = new HashSet<String>(wordList);
        Queue<String> wordQueue = new LinkedList<String>();
        wordQueue.offer(beginWord);
        int res = 0;
        while (!wordQueue.isEmpty()) {
            //先算这一层的字符串
            for (int k = wordQueue.size(); k > 0; k--) {
                String word = wordQueue.poll();
                //挺有意思，找最小，一层一层找，第一个找到的一定是最小的
                if (word.equals(endWord)) {
                    return res + 1;
                }

                for (int i = 0; i < word.length(); i++) {
                    char[] newWord = word.toCharArray();
                    //把第i个字符替换为"a~z",看哪个符合查找预期
                    for (char ch = 'a'; ch <= 'z'; ch++) {
                        newWord[i] = ch;
                        String newnewWord = new String(newWord);

                        //变换后的子串在wordList中，且还没有到达最终结果，需要继续往queue里面push
                        if (set.contains(newnewWord) && newnewWord != word) {
                            wordQueue.offer(newnewWord);
                            set.remove(newnewWord);
                        }
                    }
                }
            }
            res++;
        }
        return 0;
    }

    //https://leetcode.com/problems/game-of-life/
    //这题出的有点莫名其妙， 当前的细胞变完之后，理论上是可以影响周围的细胞变换的
    public void gameOfLife(int[][] board) {
        //-1代表过去活，现在死
        //2代表过去死，现在活
        int[] neighbors = {0, 1, -1};

        int rows = board.length;
        int cols = board[0].length;

        //遍历每一个格子里面的细胞
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {

                //统计每个细胞八个相邻位置的活细胞数量
                int liveNeighbors = 0;
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) {
                        //只算周围8个，当i,j都为0的时候，表明为
                        if (!(neighbors[i] == 0 && neighbors[j] == 0)) {
                            //相邻位置坐标
                            int r = (row + neighbors[i]);
                            int c = (col + neighbors[j]);

                            if ((r < rows && r >= 0) && (c < cols && c >= 0) && (Math.abs(board[r][c]) == 1)) {
                                liveNeighbors += 1;
                            }
                        }
                    }
                }

                //规则1或者规则3
                if ((board[row][col] == 1) && (liveNeighbors < 2 || liveNeighbors > 3)) {
                    //-1 过去是活的，现在是死的
                    board[row][col] = -1;
                }

                //规则4
                if (board[row][col] == 0 && liveNeighbors == 3) {
                    //2代表这个细胞过去是死的现在活了
                    board[row][col] = 2;
                }
            }
        }

        // 遍历 board 得到一次更新后的状态
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {

                if (board[row][col] > 0) {
                    board[row][col] = 1;
                } else {
                    board[row][col] = 0;
                }
            }
        }
    }

    //https://leetcode.com/problems/longest-consecutive-sequence/
    public int longestConsecutive(int[] nums) {
        Set<Integer> set = new HashSet<Integer>();

        for (int num : nums) {
            set.add(num);
        }

        int longestStreak = 0;

        for (int num : set) {

            //如果set里面有num-1，那么这个num一定不是最长序列的起始数字
            if (set.contains(num - 1)) {
                continue;
            }

            //找到起始数字
            int currentNum = num;
            int currentStreak = 1;

            while (set.contains(currentNum + 1)) {
                currentNum += 1;
                currentStreak += 1;
            }
            longestStreak = Math.max(longestStreak, currentStreak);
        }
        return longestStreak;
    }

    //https://leetcode.com/problems/longest-substring-without-repeating-characters/
    public int lengthOfLongestSubstring(String s) {
        Set<Character> set = new HashSet<Character>();

        int n = s.length();

        //rk为右指针
        int rk = -1, ans = 0;
        for (int i = 0; i < n; i++) {
            //左窗口右移一下
            //解释解释：只要走到for循环，就是右指针已经走完了，即右指针的下一个一定是重复字符了
            if (i != 0) {
                set.remove(s.charAt(i - 1));
            }

            //如果右指针下一个字符不在set中，则一直右移动指针
            while (rk < n - 1 && !set.contains(s.charAt(rk + 1))) {
                set.add(s.charAt(rk + 1));
                rk++;
            }

            ans = Math.max(ans, rk - i + 1);
        }
        return ans;
    }


    public int lengthOfLongestSubstring2(String s) {
        int res = 0;
        HashSet<Character> set = new HashSet<Character>();
        int start = 0;
        for (int i = 0; i < s.length(); i++) {
            char tmp = s.charAt(i);

            //hash内部一直是没有重复的
            if (!set.contains(tmp)) {
                res = Math.max(res, i - start + 1);
                set.add(tmp);
                continue;
            }

            while (start <= i) {

                char tmp2 = s.charAt(start);
                start++;
                if (tmp2 == tmp) {
                    break;
                }
                set.remove(tmp2);
            }
        }
        return res;
    }

    //https://leetcode.com/problems/number-of-islands/
    public int numIslands(char[][] grid) {
        int rows = grid.length;
        if (rows <= 0) {
            return 0;
        }
        int cols = grid[0].length;
        int[][] visited = new int[rows][cols];
        int res = 0;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == '1' && visited[i][j] == 0) {
                    res++;
                    numIslandsHelper(grid, visited, i, j);
                }
            }
        }
        return res;
    }

    public void numIslandsHelper(char[][] grid, int[][] visited, int i, int j) {

        if (i < 0 || i > grid.length - 1) {
            return;
        }

        if (j < 0 || j > grid[0].length - 1) {
            return;
        }

        if (grid[i][j] == '0' || visited[i][j] == 1) {
            return;
        }
        visited[i][j] = 1;

        numIslandsHelper(grid, visited, i + 1, j);
        numIslandsHelper(grid, visited, i - 1, j);
        numIslandsHelper(grid, visited, i, j + 1);
        numIslandsHelper(grid, visited, i, j - 1);
    }

    //https://leetcode.com/problems/binary-tree-right-side-view/
    public List<Integer> rightSideView(TreeNode root) {
        Queue<TreeNode> queue = new LinkedList<TreeNode>();
        List<Integer> res = new ArrayList<Integer>();
        if (root == null) {
            return res;
        }
        queue.offer(root);
        while (!queue.isEmpty()) {

            int size = queue.size();
            for (int i = size; i > 0; i--) {
                TreeNode tmp = queue.poll();

                if (i == size) {
                    res.add(tmp.val);
                }

                if (tmp.right != null) {
                    queue.offer(tmp.right);
                }

                if (tmp.left != null) {
                    queue.offer(tmp.left);
                }
            }
        }
        return res;
    }

    //https://leetcode.com/problems/house-robber/
    public int rob(int[] nums) {
        int len = nums.length;
        int res = 0;
        int[] dp = new int[len];

        if (len == 0) {
            return res;
        }

        if (len == 1) {
            return nums[0];
        }
        dp[0] = nums[0];
        dp[1] = Math.max(dp[0], nums[1]);
        res = Math.max(dp[1], dp[0]);
        for (int i = 2; i < len; i++) {
            dp[i] = Math.max(dp[i - 2] + nums[i], dp[i - 1]);
            res = Math.max(dp[i], res);
        }
        return res;
    }

    //https://leetcode.com/problems/repeated-dna-sequences/
    public List<String> findRepeatedDnaSequences(String s) {
        List<String> res = new ArrayList<String>();
        return res;
    }

    //https://leetcode.com/problems/reverse-words-in-a-string/
    //对于有多个空格的，有问题~  需要重新算一下
    public String reverseWords(String s) {
        s = s.trim();
        char[] arr = s.toCharArray();
        reverseHelper(arr, 0, arr.length - 1);
        int start = 0, end = 0;
        while (end < arr.length) {
            if (arr[end] != ' ') {
                end++;
                continue;
            }

            reverseHelper(arr, start, end - 1);
            start = end + 1;
            end++;
        }
        reverseHelper(arr, start, end - 1);
        return new String(arr);
    }

    public void reverseHelper(char[] input, int start, int end) {
        while (start <= end) {
            swap(input, start++, end--);
        }
        return;
    }

    //https://leetcode.com/problems/largest-number/
    public String largestNumber(int[] nums) {
        Integer[] ints = new Integer[nums.length];
        for (int i = 0; i < nums.length; i++) {
            ints[i] = nums[i];
        }

        Arrays.sort(ints, new Comparator<Integer>() {
            public int compare(Integer a, Integer b) {
                String str1 = String.valueOf(a);
                String str2 = String.valueOf(b);
                return (str2 + str1).compareTo(str1 + str2);
            }
        });

        if (ints[0] == 0) {
            return "0";
        }

        StringBuilder sb = new StringBuilder();

        for (int i = 0; i < nums.length; i++) {
            sb.append(ints[i]);
        }
        return new String(sb);
    }

    //https://leetcode.com/problems/factorial-trailing-zeroes/
    public int trailingZeroes(int n) {
        return n < 5 ? 0 : n / 5 + trailingZeroes(n / 5);
    }

    //https://leetcode.com/problems/majority-element/
    //[2,2,1,3,1,1,4,1,1,5,1,1,6]
    public int majorityElement3(int[] nums) {
        int res = nums[0];
        int cnt = 0;
        for (int i = 0; i < nums.length; i++) {
            int tmp = nums[i];
            if (tmp == res) {
                cnt++;
            } else {
                cnt--;
            }
            if (cnt <= 0) {
                res = tmp;
                cnt = 1;
            }
        }
        return res;
    }

    //https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/
    public int[] twoSum2(int[] numbers, int target) {
        int l = 0, r = numbers.length - 1;
        while (l < r) {
            if (numbers[l] + numbers[r] < target) {
                l++;
            } else if (numbers[l] + numbers[r] > target) {
                r--;
            } else {
                break;
            }
        }
        return new int[]{l + 1, r + 1};
    }

    //https://leetcode.com/problems/fraction-to-recurring-decimal/
    public String fractionToDecimal(int numerator, int denominator) {
        if (numerator == 0) {
            return "0";
        }
        StringBuilder fraction = new StringBuilder();

        //只有当一个为正，一个为负
        //true ^ false = true
        if (numerator < 0 ^ denominator < 0) {
            fraction.append("-");
        }

        //转化成long
        long dividend = Math.abs(Long.valueOf(numerator));
        long divisor = Math.abs(Long.valueOf(denominator));

        //小数点前的那一位，直接用除法可以得到
        fraction.append(String.valueOf(dividend / divisor));
        long remainder = dividend % divisor;

        //除完余数为0，直接返回，不用计算小数点后面的值
        if (remainder == 0) {
            return fraction.toString();
        }
        fraction.append(".");
        Map<Long, Integer> map = new HashMap<>();
        while (remainder != 0) {
            if (map.containsKey(remainder)) {
                fraction.insert(map.get(remainder), "(");
                fraction.append(")");
                break;
            }
            map.put(remainder, fraction.length());
            remainder *= 10;
            fraction.append(String.valueOf(remainder / divisor));
            remainder %= divisor;
        }
        return fraction.toString();
    }

    //https://leetcode.com/problems/compare-version-numbers/
    public int compareVersion(String version1, String version2) {
        String[] str1 = version1.split("\\.");
        String[] str2 = version2.split("\\.");
        int i = 0;
        while (i < str1.length || i < str2.length) {
            int s1 = i < str1.length ? Integer.valueOf(str1[i]) : 0;
            int s2 = i < str2.length ? Integer.valueOf(str2[i]) : 0;
            if (s1 < s2) {
                return -1;
            } else if (s1 > s2) {
                return 1;
            }
            i++;
        }
        return 0;
    }

    //https://leetcode.com/problems/maximum-gap/
    public int maximumGap(int[] nums) {
        if (nums.length <= 1) {
            return 0;
        }

        int max = Integer.MIN_VALUE, min = Integer.MAX_VALUE, n = nums.length, pre = 0, res = 0;
        for (int num : nums) {
            max = Integer.max(max, num);
            min = Integer.min(min, num);
        }

        //bucket的大小
        int size = (max - min) / n + 1;

        //一共有多少个bucket
        int cnt = (max - min) / size + 1;

        int[] bucketMin = new int[cnt];
        int[] bucketMax = new int[cnt];

        Arrays.fill(bucketMin, Integer.MAX_VALUE);
        Arrays.fill(bucketMax, Integer.MIN_VALUE);

        //计算每个bucket内部的最大值和最小值
        for (int num : nums) {
            int idx = (num - min) / size;
            bucketMin[idx] = Math.min(bucketMin[idx], num);
            bucketMax[idx] = Math.max(bucketMax[idx], num);
        }

        for (int i = 1; i < cnt; i++) {
            if (bucketMin[i] == Integer.MAX_VALUE || bucketMax[i] == Integer.MIN_VALUE) {
                continue;
            }

            res = Math.max(res, bucketMin[i] - bucketMax[pre]);
            pre = i;
        }

        return res;
    }

    //https://www.cnblogs.com/grandyang/p/5184890.html
    public List<String> findMissingRanges(int[] nums, int lower, int upper) {
        List<String> res = new ArrayList<String>();

        for (int num : nums) {
            if (lower == num) {
                lower = num + 1;
                continue;
            }

            //现在缺失low ~ num-1
            String str = "";
            if (num - 1 > lower) {
                str = lower + "->" + (num - 1);
            } else {
                str = String.valueOf(lower);
            }
            lower = num + 1;
            res.add(str);
        }

        int last = nums[nums.length - 1];
        if (upper - last > 2) {
            String str = "";
            if (upper - last == 2) {
                str = String.valueOf(last + 1);
            } else {
                str = (last + 1) + "->" + upper;
            }
            res.add(str);
        }
        return res;
    }


    //https://www.cnblogs.com/grandyang/p/5185561.html
    //https://leetcode.com/problems/longest-substring-with-at-most-two-distinct-characters/
    public int lengthOfLongestSubstringTwoDistinct(String s) {
        int res = 0, left = 0;
        HashMap<Character, Integer> hash = new HashMap<Character, Integer>();
        for (int i = 0; i < s.length(); i++) {
            Character tmp = s.charAt(i);
            hash.put(tmp, hash.getOrDefault(tmp, 0) + 1);
            while (hash.size() > 2) {
                Character leftChar = s.charAt(left);
                int cnt = hash.get(leftChar) - 1;

                hash.put(leftChar, cnt);

                if (cnt == 0) {
                    hash.remove(leftChar);
                }
                left++;
            }
            res = Math.max(res, i - left + 1);
        }
        return res;
    }


    public int minDistance2(String word1, String word2) {
        int len1 = word1.length();
        int len2 = word2.length();
        int[][] dp = new int[len2 + 1][len1 + 1];
        for (int i = 0; i <= len1; i++) {
            dp[0][i] = i;
        }

        for (int i = 0; i <= len2; i++) {
            dp[i][0] = i;
        }

        for (int i = 1; i <= len2; i++) {
            for (int j = 1; j <= len1; j++) {
                char s1Char = word1.charAt(j - 1);
                char s2Char = word2.charAt(i - 1);

                if (s1Char == s2Char) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = Math.min(Math.min(dp[i - 1][j - 1], dp[i - 1][j]), dp[i][j - 1]) + 1;
                }
            }
        }
        return dp[len2][len1];
    }

    //https://www.cnblogs.com/grandyang/p/4344107.html
    //https://leetcode.com/problems/one-edit-distance/
    public Boolean isOneEditDistance(String s, String t) {
        int sLen = s.length();
        int tLen = t.length();

        if ((Math.abs(sLen - tLen) > 1)) {
            return false;
        }

        if (sLen > tLen) {
            return isOneEditDistance(s.substring(0, sLen - 1), t);
        } else if (sLen < tLen) {
            return isOneEditDistance(s, t.substring(0, t.length() - 1));
        }
        return s.equals(t) ? true : isOneEditDistance(s.substring(0, sLen - 1), t.substring(0, t.length() - 1));
    }

    //https://leetcode.com/problems/evaluate-reverse-polish-notation/
    public int evalRPN(String[] tokens) {
        Stack<String> stack = new Stack<String>();
        for (String str : tokens) {
            if (str.equals("+") || str.equals("-") || str.equals("*") || str.equals("/")) {
                int i2 = Integer.valueOf(stack.pop());
                int i1 = Integer.valueOf(stack.pop());
                int tmp = 0;
                switch (str) {
                    case "+":
                        tmp = i1 + i2;
                        break;
                    case "-":
                        tmp = i1 - i2;
                        break;
                    case "*":
                        tmp = i1 * i2;
                        break;
                    case "/":
                        tmp = i1 / i2;
                        break;
                }
                stack.push(String.valueOf(tmp));
            } else {
                stack.push(str);
            }
        }
        return Integer.valueOf(stack.pop());
    }

    //https://leetcode.com/problems/sort-list/
    public ListNode sortList(ListNode head) {
        if(head == null || head.next == null){
            return head;
        }
        ListNode slow = head;
        ListNode fast = head;
        while (fast.next != null && fast.next.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        ListNode fastHead = head;
        ListNode slowHead = slow.next;

        slow.next = null;

        fastHead = sortList(fastHead);
        slowHead = sortList(slowHead);
        return mergeList(fastHead, slowHead);
    }

    //merge 两个有序链表
    public ListNode mergeList(ListNode list1, ListNode list2) {

        ListNode res = new ListNode(-1);
        ListNode head = res;

        while (list1 != null && list2 != null) {

            if (list1.val > list2.val) {
                head.next = new ListNode(list2.val);
                list2 = list2.next;
            } else {
                head.next = new ListNode(list1.val);
                list1 = list1.next;
            }
            head = head.next;
        }

        if (list1 != null) {
            head.next = list1;
        }

        if (list2 != null) {
            head.next = list2;
        }
        return res.next;
    }

}


// Definition for a Node.
class Node {
    public int val;
    public Node left;
    public Node right;
    public Node next;

    public Node() {
    }

    public Node(int _val) {
        val = _val;
    }

    public Node(int _val, Node _left, Node _right, Node _next) {
        val = _val;
        left = _left;
        right = _right;
        next = _next;
    }
}

