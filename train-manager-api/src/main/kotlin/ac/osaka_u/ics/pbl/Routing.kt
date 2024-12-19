package ac.osaka_u.ics.pbl

import ac.osaka_u.ics.pbl.handler.AssignmentsHandler
import ac.osaka_u.ics.pbl.handler.SampleHandler
import ac.osaka_u.ics.pbl.handler.TasksHandler
import ac.osaka_u.ics.pbl.model.Memo
import ac.osaka_u.ics.pbl.model.PostGeneratorRequest
import ac.osaka_u.ics.pbl.model.*
import io.ktor.http.*
import io.ktor.resources.*
import io.ktor.server.application.*
import io.ktor.server.auth.*
import io.ktor.server.request.*
import io.ktor.server.resources.*
import io.ktor.server.resources.post
import io.ktor.server.response.*
import io.ktor.server.routing.*
import org.koin.ktor.ext.inject

import java.util.concurrent.ConcurrentLinkedQueue

val taskQueue = ConcurrentLinkedQueue<Task>()
val assignedTasks = mutableMapOf<String, Task>()

//@Resource("/assignments")
//class SampleResource{
//    @Resource("next")
//    class NextResponse(val parent: SampleResource = SampleResource()) {
//    @Resource("{id}")
//        class MemoId(val parent: Memos = Memos(), val id: Int)
//    }
//}

@Resource("/assignments")
class AssignmentResource {
    @Resource("next")
    class Next(val parent: AssignmentResource = AssignmentResource())

    @Resource("{id}/refresh")
    class Refresh(val id: String, val parent: AssignmentResource = AssignmentResource())

    @Resource("{id}/register")
    class Register(val id: String, val parent: AssignmentResource = AssignmentResource())
}


//fun Application.configureRouting() {
//    val sampleHandler = SampleHandler()
//    routing {
//        get<SampleResource> {
//            call.respond(sampleHandler.handleGet())
//        }
//        get<SampleResource.NextResponse> {
//            call.respond(sampleHandler.handleGetMemos())
//        }
//        get<SampleResource.Memos.MemoId> { memoId ->
//            call.respond(sampleHandler.handleGetMemo(memoId.id))
//        }
//        post<SampleResource.Memos> {
//            val memo = call.receive<Memo>()
//            sampleHandler.handlePostMemo(memo)
//            call.respond(HttpStatusCode.Created)
//        }
//
//    }


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
    val tasksHandler by inject<TasksHandler>()
    val assignmentsHandler by inject<AssignmentsHandler>()
    routing {
        authenticate {
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

        // /assignments/next
        get<AssignmentResource.Next> {
            val userId = call.request.queryParameters["user_id"] ?: return@get call.respond(HttpStatusCode.BadRequest, "Missing user_id")
            call.respond(assignmentsHandler.assignmentNext(userId))
            val clientId = call.principal<UserIdPrincipal>()!!.name
        }

        // /assignments/{id}/refresh
        get<AssignmentResource.Refresh> { params ->
            val taskId = params.id
            val userId = call.request.queryParameters["user_id"] ?: return@get call.respond(HttpStatusCode.BadRequest, "Missing user_id")

            val task = assignedTasks[taskId]
            if (task == null) {
                call.respond(HttpStatusCode.NotFound, "Task not found")
            } else if (!isTaskAssignedToUser(taskId, userId)) {
                call.respond(HttpStatusCode.Forbidden, "Task not assigned to this user")
            } else if (!isTaskValid(task)) {
                call.respond(HttpStatusCode.Forbidden, "Task is no longer valid")
            } else {
                assignedTasks[taskId] = task.copy(completed = true)
                call.respond(task)
            }
        }

        // /assignments/{id}/register
        post<AssignmentResource.Register> { params ->
            val taskId = params.id
            val userId = call.request.queryParameters["user_id"] ?: return@post call.respond(HttpStatusCode.BadRequest, "Missing user_id")

            val task = assignedTasks[taskId]
            if (task == null) {
                call.respond(HttpStatusCode.NotFound, "Task not found")
            } else if (!isTaskAssignedToUser(taskId, userId)) {
                call.respond(HttpStatusCode.Forbidden, "Task not assigned to this user")
            } else if (!isTaskValid(task)) {
                call.respond(HttpStatusCode.Forbidden, "Task is no longer valid")
            } else {
                saveModelFile(task)
                call.respond(HttpStatusCode.OK, "Task registered successfully")
            }
        }
    }
}


    fun isTaskAssignedToUser(taskId: String, userId: String): Boolean {
        return assignedTasks[taskId]?.id == taskId
    }


    fun isTaskValid(task: Task): Boolean {
        // 示例逻辑：仅检查任务未完成状态
        return !task.completed
    }


    fun generateTask(): Task? {
        return Task(
            id = System.currentTimeMillis().toString(),
            completed = false,
            base_model_id = "base123",
            type = "type1",
            parameter = Parameter1(player_id = "player1", game_id = listOf("game1", "game2"))
        )
    }

    fun saveModelFile(task: Task) {
        println("Saving model file for task: ${task.id}")
    }

