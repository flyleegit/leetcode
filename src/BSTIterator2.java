import jdk.jfr.DataAmount;

import java.util.Stack;

public class BSTIterator2 {

    Stack<TreeNode> stack;

    public BSTIterator2(TreeNode root) {
        this.stack = new Stack<TreeNode>();
        this._leftmostInorder(root);
    }

    private void _leftmostInorder(TreeNode root) {
        while (root != null) {
            this.stack.push(root);
            root = root.left;
        }
    }

    /**
     * @return the next smallest number
     */
    public int next() {
        TreeNode topmostNode = this.stack.pop();

        if (topmostNode.right != null) {
            this._leftmostInorder(topmostNode.right);
        }
        return topmostNode.val;
    }

    /**
     * @return whether we have a next smallest number
     */
    public boolean hasNext() {
        return this.stack.size() > 0;
    }
}
/**
 * Your BSTIterator object will be instantiated and called as such:
 * BSTIterator obj = new BSTIterator(root);
 * int param_1 = obj.next();
 * boolean param_2 = obj.hasNext();
 */
