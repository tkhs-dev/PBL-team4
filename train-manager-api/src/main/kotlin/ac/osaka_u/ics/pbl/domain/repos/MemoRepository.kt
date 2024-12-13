package ac.osaka_u.ics.pbl.domain.repos

import ac.osaka_u.ics.pbl.model.Memo

class MemoRepository {
    val memos = mutableMapOf(1 to Memo("Hello, world!"))
    fun getMemo(id: Int): Memo? {
        return memos[id]
    }

    fun getMemos(): List<Memo> {
        return memos.values.toList()
    }

    fun postMemo(post: Memo) {
        memos[memos.size + 1] = post
    }
}