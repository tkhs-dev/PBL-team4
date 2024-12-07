package ac.osaka_u.ics.pbl

import ac.osaka_u.ics.pbl.handler.SampleHandler
import ac.osaka_u.ics.pbl.handler.TasksHandler
import ac.osaka_u.ics.pbl.model.Memo
import ac.osaka_u.ics.pbl.model.PostGeneratorRequest
import io.ktor.http.*
import io.ktor.resources.*
import io.ktor.server.application.*
import io.ktor.server.request.*
import io.ktor.server.resources.*
import io.ktor.server.resources.post
import io.ktor.server.response.*
import io.ktor.server.routing.*

@Resource("/sample")
class SampleResource{
    @Resource("memos")
    class Memos(val parent: SampleResource = SampleResource()) {
        @Resource("{id}")
        class MemoId(val parent: Memos = Memos(), val id: Int)
    }
}

@Resource("/tasks")
class TaskResource{
    @Resource("{id}")
    class TaskId(val parent: TaskResource = TaskResource(), val id: String)
    @Resource("generators")
    class Generators(val parent: TaskResource = TaskResource()){
        @Resource("{id}")
        class GeneratorId(val parent: Generators = Generators(), val id: Int)
    }
}

fun Application.configureRouting() {
    val sampleHandler = SampleHandler()
    val tasksHandler = TasksHandler()
    routing {
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

        get<TaskResource> {
            call.respond(tasksHandler.handleGetTasks())
        }
        get<TaskResource.TaskId> { taskId ->
            call.respond(tasksHandler.handleGetTask(taskId.id))
        }
        get<TaskResource.Generators> {
            call.respond(tasksHandler.handleGetGenerators())
        }
        post<TaskResource.Generators> {
            val request = call.receive<PostGeneratorRequest>()
            val resp = tasksHandler.handlePostGenerator(request)
            call.respond(HttpStatusCode.Created, resp)
        }
        get<TaskResource.Generators.GeneratorId> { generatorId ->
            call.respond(tasksHandler.handleGetGenerator(generatorId.id))
        }
        delete<TaskResource.Generators.GeneratorId> { generatorId ->
            tasksHandler.handleDeleteGenerator(generatorId.id)
            call.respond(HttpStatusCode.NoContent)
        }
    }
}
