package ac.osaka_u.ics.pbl

import ac.osaka_u.ics.pbl.handler.SampleHandler
import ac.osaka_u.ics.pbl.model.Memo
import io.ktor.http.*
import io.ktor.resources.*
import io.ktor.server.application.*
import io.ktor.server.request.*
import io.ktor.server.resources.*
import io.ktor.server.response.*
import io.ktor.server.routing.routing

@Resource("/sample")
class SampleResource{
    @Resource("memos")
    class Memos(val parent: SampleResource = SampleResource()) {
        @Resource("{id}")
        class MemoId(val parent: Memos = Memos(), val id: Int)
    }
}

fun Application.configureRouting() {
    val sampleHandler = SampleHandler()
    routing {
        get<SampleResource> {
            call.respond(sampleHandler.handleGet())
        }
        get<SampleResource.Memos> {
            call.respond(sampleHandler.handleGetMemos())
        }
        get<SampleResource.Memos.MemoId> { memoId ->
            call.respond(sampleHandler.handleGetMemo(memoId.id))
        }
        post<SampleResource.Memos> {
            val memo = call.receive<Memo>()
            sampleHandler.handlePostMemo(memo)
            call.respond(HttpStatusCode.Created)
        }
    }
}
