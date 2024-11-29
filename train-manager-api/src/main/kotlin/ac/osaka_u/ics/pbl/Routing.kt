package ac.osaka_u.ics.pbl

import io.ktor.resources.*
import io.ktor.server.application.*
import io.ktor.server.resources.*
import io.ktor.server.response.*
import io.ktor.server.routing.*
import io.ktor.server.routing.get

@Resource("/sample")
class SampleResource{
    @Resource("/memo")
    class Memo
}


fun Application.configureRouting() {
    routing {
    }
}
