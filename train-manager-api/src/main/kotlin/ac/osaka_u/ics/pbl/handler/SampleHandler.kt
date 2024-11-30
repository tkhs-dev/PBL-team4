package ac.osaka_u.ics.pbl.handler

import ac.osaka_u.ics.pbl.ApiException
import ac.osaka_u.ics.pbl.model.Memo
import ac.osaka_u.ics.pbl.repos.MemoRepository
import kotlinx.serialization.Serializable

@Serializable
data class TextMessage(val text: String)

class SampleHandler {
    private val memoRepository = MemoRepository()

    fun handleGet() : TextMessage {
        return TextMessage("Hello, world!")
    }

    fun handleGetMemo(id: Int): Memo {
        return memoRepository.getMemo(id) ?: throw ApiException.NotFoundException()
    }

    fun handleGetMemos(): List<Memo> {
        return memoRepository.getMemos()
    }

    fun handlePostMemo(memo: Memo) {
        return memoRepository.postMemo(memo)
    }
}